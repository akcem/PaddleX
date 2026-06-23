import sys
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parents[1]

DET_CONFIG = (
    REPO_ROOT
    / "paddlex"
    / "configs"
    / "modules"
    / "text_detection"
    / "PP-OCRv5_server_det.yaml"
)
REC_CONFIG = (
    REPO_ROOT
    / "paddlex"
    / "configs"
    / "modules"
    / "text_recognition"
    / "PP-OCRv5_server_rec.yaml"
)
RUNTIME_CONFIG_DIR = REPO_ROOT / "mytools" / "runtime_configs"


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def default_python():
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def norm_path(path):
    return str(Path(str(path).strip().strip('"')).expanduser())
