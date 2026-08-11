"""Persistent, extensible storage for user-supplied test assets."""

from __future__ import annotations

import logging
import re
from pathlib import Path


logger = logging.getLogger(__name__)


class AssetManager:
    """Store one current asset per logical asset type.

    Assets are deliberately addressed by a stable type (``resume``,
    ``profile_image``, ``invoice``) rather than by UI-specific filenames.  New
    asset types can therefore use the same save and resolution APIs.
    """

    _ASSET_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
    RESUME_EXTENSIONS = {".pdf", ".doc", ".docx"}

    def __init__(self, upload_dir: Path | None = None) -> None:
        self.upload_dir = upload_dir or Path(__file__).resolve().parent.parent / "uploads"

    def save_asset(self, asset_type: str, filename: str, content: bytes) -> Path:
        """Replace the current asset of ``asset_type`` and return its path."""
        self._validate_asset_type(asset_type)
        extension = Path(filename).suffix.lower()
        if not extension:
            raise ValueError("Uploaded asset must include a file extension")

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        for existing in self.upload_dir.glob(f"{asset_type}.*"):
            if existing.is_file():
                existing.unlink()

        asset_path = self.upload_dir / f"{asset_type}{extension}"
        asset_path.write_bytes(content)
        logger.info("Saved test asset '%s' to %s", asset_type, asset_path)
        return asset_path

    def get_asset_path(self, asset_type: str) -> Path | None:
        """Return the stored path for an asset type, if it exists."""
        self._validate_asset_type(asset_type)
        matches = [path for path in self.upload_dir.glob(f"{asset_type}.*") if path.is_file()]
        if not matches:
            logger.warning("No stored test asset found for '%s'", asset_type)
            return None
        asset_path = max(matches, key=lambda path: path.stat().st_mtime)
        logger.info("Resolved test asset '%s' to %s", asset_type, asset_path)
        return asset_path

    def list_assets(self) -> dict[str, Path]:
        """All stored assets by type — what upload test steps may reference."""
        if not self.upload_dir.is_dir():
            return {}
        assets: dict[str, Path] = {}
        for path in sorted(self.upload_dir.iterdir()):
            if path.is_file() and self._ASSET_TYPE_PATTERN.fullmatch(path.stem):
                assets[path.stem] = path
        return assets

    def save_resume(self, filename: str, content: bytes) -> Path:
        """Validate and persist the resume asset."""
        if Path(filename).suffix.lower() not in self.RESUME_EXTENSIONS:
            raise ValueError("Resume must be a .pdf, .doc, or .docx file")
        return self.save_asset("resume", filename, content)

    def resolve_resume_path(self) -> Path | None:
        """Resolve the resume asset for a browser file upload."""
        return self.get_asset_path("resume")

    @classmethod
    def _validate_asset_type(cls, asset_type: str) -> None:
        if not cls._ASSET_TYPE_PATTERN.fullmatch(asset_type):
            raise ValueError(f"Invalid asset type: {asset_type!r}")
