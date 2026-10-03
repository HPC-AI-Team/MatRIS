"""Model namespace; lazy import lets graph construction use shared operators."""
__all__ = ["MatRIS"]


def __getattr__(name):
    if name == "MatRIS":
        from .model import MatRIS
        return MatRIS
    raise AttributeError(name)
