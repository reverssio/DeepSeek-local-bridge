"""Entry point — run the OpenAI-compatible DeepSeek server.

    python app.py            # serves on http://localhost:8000

On first use (no saved session) a browser window opens for you to sign in by
hand; run `python -m deepseek.auth` to do that ahead of time. Set
DEEPSEEK_PROFILE_DIR to reuse an existing signed-in Chrome profile.
"""

import faulthandler
import os
import signal

import uvicorn
from dotenv import load_dotenv

# SIGUSR1 -> dump all thread stacks to stderr/log (debugging aid; stacks show
# code locations only, never request payloads)
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except (AttributeError, ValueError):
    pass

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "server.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
