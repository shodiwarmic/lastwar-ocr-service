"""
main.py

Gunicorn entrypoint for the Last War OCR microservice.

Gunicorn imports this module and looks for the 'app' object. The CMD in the
Dockerfile specifies 'main:app' which resolves to the 'app' variable in this
file.

Cloud Run best practices applied here:
    - App is created at module level (not inside a function) so Gunicorn's
      worker processes share the initialised app without re-running the factory.
    - The Vision API client is NOT initialised here — it initialises lazily on
      the first request, keeping container startup time fast. Cloud Run has a
      startup timeout and slow starts can cause deployment failures.
    - PORT is read from the environment (Cloud Run sets this to 8080). The
      Gunicorn bind address in the Dockerfile already uses $PORT but this
      fallback allows local development with `python main.py`.
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Local development only — Gunicorn is used in production (see Dockerfile CMD)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
