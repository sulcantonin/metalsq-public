"""Meta-LSQ: few-shot meta-learning with a closed-form Sobolev inner solver."""
from .basis import PolynomialColumns, SineBasis
from .model import Adapted, MetaLSQ, Task, ridge_solve
from . import tasks
from .tasks import Standardizer
from . import baselines

__version__ = "0.1.0"
__all__ = ["MetaLSQ", "Adapted", "Task", "SineBasis", "PolynomialColumns", "ridge_solve", "tasks", "Standardizer", "baselines"]
