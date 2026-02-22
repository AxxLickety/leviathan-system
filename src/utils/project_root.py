from pathlib import Path

def get_project_root() -> Path:
    """
    Returns the absolute path of the project root directory.
    Works no matter where Jupyter/Python is running.
    """
    return Path(__file__).resolve().parents[2]
