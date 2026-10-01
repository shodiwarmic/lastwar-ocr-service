"""
app/version.py

The release and commit this process is running, reported on /health and in
every /process-batch diagnostics block.

Both are baked into the image at build time (the Dockerfiles turn the
APP_VERSION / APP_COMMIT build args into these environment variables); a
source checkout run by hand reports `dev` / `unknown`.
"""

import os

SERVICE_VERSION = os.environ.get("OCR_SERVICE_VERSION", "") or "dev"
SERVICE_COMMIT = os.environ.get("OCR_SERVICE_COMMIT", "") or "unknown"
