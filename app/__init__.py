"""
app/__init__.py

Flask application factory.

Using the factory pattern (create_app) rather than a module-level app object
allows the app to be instantiated with different configurations for testing
vs production without import-time side effects.

Cloud Run instantiates the container once and then routes multiple requests
to the same process. The factory is called once in main.py and the resulting
app object is reused across all requests.
"""

from flask import Flask
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> Flask:
    """
    Creates and configures the Flask application.

    Registers all blueprints and sets production-appropriate config values.
    The Vision API client is NOT initialised here — it is lazily initialised
    on the first request to avoid slowing down container startup time, which
    Cloud Run measures against a startup timeout.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Disable Flask's default exception propagation in production so unhandled
    # errors return a clean JSON 500 instead of an HTML traceback page
    app.config["PROPAGATE_EXCEPTIONS"] = False

    # Register route blueprints
    from app.routes import bp
    app.register_blueprint(bp)

    logger.info("Flask application created")
    return app
