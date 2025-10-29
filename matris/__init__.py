from importlib.metadata import version

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)  # read from pyproject.toml
except PackageNotFoundError:
    __version__ = "unknown"