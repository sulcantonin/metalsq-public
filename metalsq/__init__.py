"""Meta-LSQ: few-shot meta-learning with a closed-form Sobolev inner solver."""
from .basis import SineBasis
from .model import Adapted, MetaLSQ, Task, ridge_solve
from . import tasks

__version__ = "0.1.0"
__all__ = ["MetaLSQ", "Adapted", "Task", "SineBasis", "ridge_solve", "tasks"]
