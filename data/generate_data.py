"""Stand-alone entry point: python data/generate_data.py --student-id ... --seed ..."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Paths  # noqa: E402
from src.data_generation import write_dataset  # noqa: E402
from src.validation import parse_identity  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    ident = parse_identity(a.student_id, a.seed)
    paths = Paths(ROOT, ident["regdigits"])
    paths.ensure()
    print(ident["verification_code"], write_dataset(ident["seed"], paths.csv, paths.fingerprint,
                                                    ident["verification_code"]))
