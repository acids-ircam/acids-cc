import sys
from pathlib import Path

sys_path = str(Path(__file__).parent.parent)
if sys_path not in sys.path: sys.path.append(sys_path)