"""
Flask application factory.

HSA override must run before any import that loads torch. Blueprint imports are
deferred until inside create_app() after apply_hsa_override().
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from utils.device import apply_hsa_override


def create_app(test_config: dict | None = None):
    apply_hsa_override()

    from flask import Flask

    from web.routes import bp as main_bp

    pkg_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(pkg_dir / "templates"),
        static_folder=str(pkg_dir / "static"),
        static_url_path="/static",
    )

    app.config.from_mapping(
        MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024,
        UPLOAD_ROOT=pkg_dir / "uploads",
        FIGURES_ROOT=pkg_dir.parent / "figures",
        PROJECT_ROOT=pkg_dir.parent,
    )

    if test_config:
        app.config.update(test_config)

    app.register_blueprint(main_bp)

    upload_root: Path = app.config["UPLOAD_ROOT"]
    upload_root.mkdir(parents=True, exist_ok=True)

    if os.environ.get("WEB_CLEAR_UPLOADS_ON_START") == "1":
        for child in upload_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_file():
                try:
                    child.unlink()
                except OSError:
                    pass

    return app
