import os

import torch

from PyMAIA.utils.log_utils import get_logger

try:

    GPU_AVAILABLE = torch.cuda.is_available()


except:  # noqa: E722
    GPU_AVAILABLE = False

logger = get_logger("PyMAIA", "INFO")
logger.info("PyMAIA GPU Available: {}".format(GPU_AVAILABLE))
os.environ["GPU_AVAILABLE"] = str(GPU_AVAILABLE)  # noqa: F405

from . import _version

__version__ = _version.get_versions()['version']
