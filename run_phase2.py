from pathlib import Path
import yaml
from src.phase2_supply.pipeline import run

ROOT = Path(__file__).resolve().parent

def main():
    cfg_path = ROOT / "config" / "phase2" / "supply_gate.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    run(cfg, root=ROOT)

if __name__ == "__main__":
    main()
