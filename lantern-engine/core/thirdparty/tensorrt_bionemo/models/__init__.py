def __getattr__(name):
    if name == "Boltz1":
        from .boltz1 import Boltz1
        return Boltz1
    if name in ("Boltz2", "Boltz2Affinity"):
        from .boltz2 import Boltz2, Boltz2Affinity
        return locals()[name]
    if name == "OpenFold2":
        from .openfold2 import OpenFold2
        return OpenFold2
    if name == "OpenFold3":
        from .openfold3 import OpenFold3
        return OpenFold3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Boltz1", "Boltz2", "Boltz2Affinity", "OpenFold2", "OpenFold3"]
