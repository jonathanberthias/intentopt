"""Priority-based compromise optimization using raw CVXPY intentions."""

from .integrality import Integral, MultipleOf
from .model import Intentions
from .search import BinarySearch, SearchPolicy, SequentialSearch
from .solution import IntentSolution, SolveAttempt, Violation
from .solve import solve

__version__ = "0.1.0"

__all__ = [
    "BinarySearch",
    "Integral",
    "IntentSolution",
    "Intentions",
    "MultipleOf",
    "SearchPolicy",
    "SequentialSearch",
    "SolveAttempt",
    "Violation",
    "__version__",
    "solve",
]
