"""Multimodal image pipeline for DeepSeek Web2API bridge on Android/Termux.

Supports:
- Local Android shared storage & Termux paths (/storage/emulated/0/..., ~/storage/shared/...)
- Base64 data URIs (data:image/...;base64,...)
- Remote public image URLs with comprehensive SSRF and resource bounds protection
- Automatic path detection from natural conversation prompts
- Standard OpenAI chat completion multimodal message formats
- DeepSeek Web file upload protocol (/api/v0/file/upload_file with PoW & VISION model kind)
- SHA-256 caching of uploaded file IDs across turns to minimize redundant uploads
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import logging
import os
import re
import socket
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx

logger = logging.getLogger("multimodal")

# Maximum image size allowed for ingestion (10 MB per Android/Termux constraints)
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

# Allowed local filesystem roots (Android, Termux, and standard desktop/server environments)
_DEFAULT_ALLOWED_ROOTS = [
    "/storage/emulated/0",
    "/data/data/com.termux/files",
    "/sdcard",
]


def get_allowed_local_roots() -> list[str]:
    """Dynamically resolve allowed storage roots across Android and desktop platforms."""
    roots = list(_DEFAULT_ALLOWED_ROOTS)
    try:
        roots.append(str(Path.home().resolve()))
    except Exception:
        pass
    try:
        roots.append(str(Path.cwd().resolve()))
    except Exception:
        pass
    env_roots = os.environ.get("MULTIMODAL_ALLOWED_ROOTS", "")
    if env_roots:
        for r in re.split(r"[:;]", env_roots):
            if r.strip():
                roots.append(str(Path(r.strip()).resolve()))
    return [r for r in roots if os.path.exists(r)]

ALLOWED_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# Image magic bytes
MAGIC_NUMBERS = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"BM": "image/bmp",
}

# Regex to detect natural local paths in prompt text
LOCAL_PATH_RE = re.compile(
    r'(?:(?:/storage/emulated/0/|/data/data/com\.termux/files/|/sdcard/|~/)[^\s"\'<>()]+?\.(?:jpg|jpeg|png|webp|gif|bmp))',
    re.I
)

# Regex to detect HTTP/HTTPS image URLs in prompt text
IMAGE_URL_RE = re.compile(
    r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif|bmp)(?:\?[^\s"\'<>]*)?',
    re.I
)

# Cache of image hash -> deepseek file_id
_uploaded_file_id_cache: Dict[str, str] = {}


@dataclass
class ImageAsset:
    data: bytes
    filename: str
    mime_type: str
    source_type: str  # "local", "remote", "data_uri"
    location: str


def detect_mime_from_bytes(data: bytes) -> Optional[str]:
    """Inspect magic numbers to verify genuine image content."""
    for magic, mime in MAGIC_NUMBERS.items():
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def resolve_local_path(raw_path: str) -> Path:
    """Expand ~ and resolve symlinks, ensuring the path stays within allowed roots."""
    expanded = os.path.expanduser(raw_path)
    p = Path(expanded).resolve()

    resolved_str = str(p)
    allowed_roots = get_allowed_local_roots()
    is_allowed = any(resolved_str == root or resolved_str.startswith(root.rstrip("/") + "/") for root in allowed_roots)
    if not is_allowed:
        raise PermissionError(f"Access denied: path '{raw_path}' is outside allowed storage roots")
    return p


def validate_and_load_local_image(raw_path: str) -> ImageAsset:
    """Load and validate an image from the local Android/Termux filesystem."""
    p = resolve_local_path(raw_path)

    if not p.exists():
        raise FileNotFoundError(f"Image file does not exist: '{raw_path}'")
    if p.is_dir():
        raise IsADirectoryError(f"Target path is a directory, not an image file: '{raw_path}'")

    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS.keys()))}")

    size = p.stat().st_size
    if size == 0:
        raise ValueError(f"Image file is empty: '{raw_path}'")
    if size > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(f"Image exceeds size limit of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB (was {size} bytes)")

    data = p.read_bytes()
    mime = detect_mime_from_bytes(data) or ALLOWED_EXTENSIONS[ext]

    return ImageAsset(
        data=data,
        filename=p.name,
        mime_type=mime,
        source_type="local",
        location=str(p),
    )


def validate_remote_url(url: str) -> None:
    """Strict SSRF security validation for outbound image requests."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        raise ValueError(f"Malformed URL: {e}")

    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    try:
        addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except Exception as e:
        raise ConnectionError(f"DNS resolution failed for '{hostname}': {e}")

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise SecurityError(f"SSRF violation: target hostname resolves to forbidden private/loopback IP {ip_str}")
        except ValueError:
            raise SecurityError(f"Invalid resolved IP address: {ip_str}")


class SecurityError(PermissionError):
    pass


def fetch_remote_image(url: str) -> ImageAsset:
    """Safely download a remote image with SSRF checks, timeout, and size limits."""
    validate_remote_url(url)

    transport = httpx.HTTPTransport(retries=1)
    with httpx.Client(transport=transport, timeout=httpx.Timeout(10.0, read=15.0), follow_redirects=True) as client:
        # Re-verify any redirects for SSRF
        resp = client.get(url, headers={"User-Agent": "DeepSeekBridge/2.0 (Android; Termux)"})
        resp.raise_for_status()

        final_url = str(resp.url)
        validate_remote_url(final_url)

        content_length = resp.headers.get("content-length")
        if content_length and int(content_length) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Remote image exceeds size limit ({content_length} bytes)")

        data = resp.content
        if len(data) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Remote image payload exceeds size limit ({len(data)} bytes)")
        if len(data) == 0:
            raise ValueError("Remote image payload is empty")

        mime = detect_mime_from_bytes(data)
        if not mime:
            ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ct in ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"):
                mime = ct
            else:
                raise ValueError(f"Remote resource is not a recognized image format (Content-Type: {ct})")

        filename = os.path.basename(urllib.parse.urlparse(final_url).path) or f"download_{int(time.time())}.img"
        return ImageAsset(
            data=data,
            filename=filename,
            mime_type=mime,
            source_type="remote",
            location=url,
        )


def parse_data_uri(uri: str) -> ImageAsset:
    """Parse base64 data URI (data:image/...;base64,...)."""
    match = re.match(r"^data:(image/[a-zA-Z0-9_.+-]+);base64,(.+)$", uri, re.DOTALL)
    if not match:
        raise ValueError("Invalid image data URI format")

    declared_mime = match.group(1).lower()
    encoded = match.group(2).strip()
    try:
        data = base64.b64decode(encoded)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image data: {e}")

    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(f"Data URI image exceeds {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB limit")

    detected_mime = detect_mime_from_bytes(data) or declared_mime
    ext = ".jpg" if "jpeg" in detected_mime else f".{detected_mime.split('/')[-1]}"

    return ImageAsset(
        data=data,
        filename=f"embedded_{hashlib.sha256(data).hexdigest()[:8]}{ext}",
        mime_type=detected_mime,
        source_type="data_uri",
        location="data_uri",
    )


def extract_images_from_messages(messages: list) -> List[ImageAsset]:
    """Scan incoming messages for OpenAI multimodal image blocks and natural text paths."""
    assets: List[ImageAsset] = []
    seen_locations: Set[str] = set()

    for m in messages:
        # 1. Inspect dict or ChatMessage content
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)

        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "image_url":
                        img_obj = part.get("image_url", {})
                        url = img_obj.get("url") if isinstance(img_obj, dict) else img_obj
                        if isinstance(url, str) and url not in seen_locations:
                            seen_locations.add(url)
                            if url.startswith("data:"):
                                assets.append(parse_data_uri(url))
                            elif url.startswith("http://") or url.startswith("https://"):
                                assets.append(fetch_remote_image(url))
                            else:
                                # treat as local file path
                                assets.append(validate_and_load_local_image(url))
                    elif ptype == "text" and isinstance(part.get("text"), str):
                        text = part["text"]
                        for match in LOCAL_PATH_RE.findall(text):
                            if match not in seen_locations:
                                seen_locations.add(match)
                                try:
                                    assets.append(validate_and_load_local_image(match))
                                except Exception as e:
                                    logger.warning("Could not load detected local path '%s': %s", match, e)

        elif isinstance(content, str):
            # Check for natural local file paths mentioned in user text
            for match in LOCAL_PATH_RE.findall(content):
                if match not in seen_locations:
                    seen_locations.add(match)
                    try:
                        assets.append(validate_and_load_local_image(match))
                    except Exception as e:
                        logger.warning("Could not load detected local path '%s': %s", match, e)

            # Check for explicit standalone image URLs in user text
            for match in IMAGE_URL_RE.findall(content):
                if match not in seen_locations:
                    seen_locations.add(match)
                    try:
                        assets.append(fetch_remote_image(match))
                    except Exception as e:
                        logger.warning("Could not fetch detected remote image URL '%s': %s", match, e)

    return assets


def upload_image_to_deepseek(client, asset: ImageAsset, timeout: float = 30.0) -> str:
    """Upload an image to DeepSeek Web /api/v0/file/upload_file and poll until READY.

    Returns the upstream file_id for inclusion in ref_file_ids.
    """
    img_hash = hashlib.sha256(asset.data).hexdigest()
    if img_hash in _uploaded_file_id_cache:
        cached_id = _uploaded_file_id_cache[img_hash]
        logger.info("Reusing cached DeepSeek file_id for image %s: %s", asset.filename, cached_id)
        return cached_id

    logger.info("Uploading image to DeepSeek Web (%s, %d bytes)...", asset.filename, len(asset.data))

    # 1. Request and solve PoW for target path '/api/v0/file/upload_file'
    pow_header = client._pow_header(target_path="/api/v0/file/upload_file")

    files = {"file": (asset.filename, asset.data, asset.mime_type)}
    headers = {"x-ds-pow-response": pow_header}

    # Temporarily remove content-type from client._http.headers so httpx adds correct multipart boundary
    old_ct = client._http.headers.pop("content-type", None)
    try:
        resp = client._http.post("/api/v0/file/upload_file", files=files, headers=headers)
        resp.raise_for_status()
        res_json = resp.json()
    finally:
        if old_ct:
            client._http.headers["content-type"] = old_ct

    if res_json.get("code") != 0 or res_json.get("data", {}).get("biz_code") != 0:
        err_msg = res_json.get("msg") or res_json.get("data", {}).get("biz_msg") or f"Error code {res_json.get('code')}"
        raise RuntimeError(f"DeepSeek file upload failed: {err_msg}")

    biz_data = res_json.get("data", {}).get("biz_data") or {}
    file_id = biz_data.get("id")
    if not file_id:
        raise RuntimeError(f"DeepSeek upload response missing file id: {res_json}")

    # 2. Poll file status via /api/v0/file/fetch_files?file_ids=<id>
    start_time = time.time()
    while time.time() - start_time < timeout:
        time.sleep(1.0)
        poll_resp = client._http.get("/api/v0/file/fetch_files", params={"file_ids": file_id})
        poll_resp.raise_for_status()
        poll_json = poll_resp.json()
        files_list = poll_json.get("data", {}).get("biz_data", {}).get("files", [])
        if files_list:
            status = files_list[0].get("status")
            if status == "SUCCESS":
                _uploaded_file_id_cache[img_hash] = file_id
                logger.info("Image upload confirmed ready (file_id=%s)", file_id)
                return file_id
            elif status in ("FAILED", "ERROR"):
                err_code = files_list[0].get("error_code")
                raise RuntimeError(f"DeepSeek failed to parse uploaded image (status={status}, code={err_code})")

    # If polling timed out, return file_id best-effort
    logger.warning("DeepSeek file status polling timed out after %ds; proceeding with file_id=%s", timeout, file_id)
    _uploaded_file_id_cache[img_hash] = file_id
    return file_id


def process_multimodal_inputs_with_meta(client, messages: list) -> Tuple[List[str], List[str]]:
    """Inspect messages for images, upload each to DeepSeek Web, and return (file_ids, locations)."""
    assets = extract_images_from_messages(messages)
    if not assets:
        return [], []

    file_ids: List[str] = []
    locations: List[str] = []
    for asset in assets:
        fid = upload_image_to_deepseek(client, asset)
        file_ids.append(fid)
        locations.append(asset.location)

    return file_ids, locations


def process_multimodal_inputs(client, messages: list) -> List[str]:
    """Inspect messages for images, upload each to DeepSeek Web, and return list of file_ids."""
    file_ids, _ = process_multimodal_inputs_with_meta(client, messages)
    return file_ids
