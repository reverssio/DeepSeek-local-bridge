#!/data/data/com.termux/files/usr/bin/python
"""Discover and connect to the phone's OWN wireless-adb service via mDNS.

Android's wireless debugging publishes _adb-tls-connect._tcp.local over
mDNS on the local network. On this single-phone setup, the advertised service
IS this phone, so we can safely auto-connect to it (the device is already
paired/persistently authorized; no pairing code needed for plain 'connect'
when previously authorized).

Used by start.sh / network-test.sh to recover adb after Wi-Fi changes.
Prints human-readable status; exit 0 when a device is connected.
"""

from __future__ import annotations

import socket
import struct
import subprocess
import sys
import time


def parse_name(data: bytes, off: int):
    labels = []
    while True:
        try:
            l = data[off]
        except IndexError:
            raise ValueError("bad dns name")
        if l == 0:
            off += 1
            break
        if l & 0xC0 == 0xC0:
            ptr = struct.unpack(">H", data[off : off + 2])[0] & 0x3FFF
            name, _ = parse_name(data, ptr)
            labels.append(name)
            off += 2
            return ".".join(labels), off
        off += 1
        labels.append(data[off : off + l].decode("utf-8", "replace"))
        off += l
    return ".".join(labels), off


def mdns_query(sock: socket.socket):
    for svc in (b"_adb-tls-connect._tcp.local", b"_adbservices._tcp.local"):
        q = (
            b"\xab\xcd\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
            + bytes([len(svc)])
            + svc
            + b"\x00\x00\x0c\x00\x01"
        )
        sock.sendto(q, ("224.0.0.251", 5353))


def parse_adb_ports(data: bytes):
    """Return SRV (name, port) pairs for adb services in an mDNS packet."""
    out = []
    try:
        qdcount = struct.unpack(">H", data[4:6])[0]
        ancount = struct.unpack(">H", data[6:8])[0]
        off = 12
        for _ in range(qdcount):
            _, off = parse_name(data, off)
            off += 4
        for _ in range(ancount):
            name, off = parse_name(data, off)
            rtype, _rclass, _ttl, rlen = struct.unpack(">HHIH", data[off : off + 10])
            off += 10
            rdata = data[off : off + rlen]
            if rtype == 33 and b"adb" in name.encode() if isinstance(name, str) else False:
                pass
            if rtype == 33:  # SRV
                try:
                    port = struct.unpack(">HHH", rdata[:6])[2]
                    out.append((name, port))
                except Exception:
                    pass
            off += rlen
    except Exception:
        pass
    return out


def adb(*args):
    return subprocess.run(
        ["adb", *args], capture_output=True, text=True, timeout=30
    )


def devices() -> list:
    r = adb("devices")
    out = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "device":
            out.append(parts[0])
    return out


def main() -> int:
    # 1) already have a device?
    dev = devices()
    if dev:
        print(f"adb already connected: {', '.join(dev)}")
        return 0

    # 2) try the remembered target
    try:
        saved = open(
            "/data/data/com.termux/files/home/deepseek-api/session/adb_target.txt"
        ).read().strip()
    except FileNotFoundError:
        saved = None
    if saved:
        adb("connect", saved)
        dev = devices()
        if dev:
            print(f"adb reconnected to remembered target {saved}")
            return 0

    # 3) mDNS discovery (Wi-Fi only; works while phone is on same network)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", 5353))
        mgroup = socket.inet_aton("224.0.0.251") + socket.inet_aton("0.0.0.0")
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mgroup)
    except OSError as e:
        print(f"mDNS unavailable ({e}); trying saved target only")
        return 1
    sock.settimeout(3)
    mdns_query(sock)
    end = time.time() + 8
    seen = []
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            mdns_query(sock)
            continue
        if b"adb" not in data.lower():
            continue
        for name, port in parse_adb_ports(data):
            if (name, port) not in seen:
                seen.append((name, port))
        if seen:
            break
    if not seen:
        print("no adb mDNS service found (is Wireless debugging on?)")
        return 1

    # Connect to the advertised service (this phone).
    host = addr[0]
    for name, port in seen:
        adb("connect", f"{host}:{port}")
        dev = devices()
        if dev:
            print(f"adb connected via mDNS to {host}:{port} ({name})")
            try:
                with open(
                    "/data/data/com.termux/files/home/deepseek-api/session/adb_target.txt",
                    "w",
                ) as f:
                    f.write(f"{host}:{port}\n")
            except OSError:
                pass
            return 0
    print("mDNS found a service but adb connect failed (may need re-pairing)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
