import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(root)
sys.path.append(parent)

from bikeshare_model.config.core import PACKAGE_ROOT, config

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()