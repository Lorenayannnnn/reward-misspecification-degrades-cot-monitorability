"""
One module per task, each registering its Task, its two rewards and its hack label under the
names the configs use. Importing the package registers all three tasks.
"""

from src.tasks import coding  # noqa: F401
from src.tasks import length  # noqa: F401
from src.tasks import judge  # noqa: F401
