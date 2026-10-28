"""SeedLabeler: stereo seed traits, outline review and per-crop model training."""
from importlib.metadata import version as _v

try:
    __version__ = _v("seedlabeler")
except Exception:
    __version__ = "0.0.0"
