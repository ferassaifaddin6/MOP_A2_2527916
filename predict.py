"""CLI inference: python predict.py --json samples/valid.json"""
import argparse
import json
import sys

from pydantic import ValidationError

from src.inference import load_bundle, predict_record

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to one JSON farm record")
    a = ap.parse_args()
    try:
        record = json.load(open(a.json))
        print(json.dumps(predict_record(record, load_bundle()), indent=2))
    except ValidationError as exc:
        errs = [{"field": ".".join(map(str, e["loc"])), "problem": e["msg"]} for e in exc.errors()]
        print(json.dumps({"error": "invalid input", "details": errs}, indent=2), file=sys.stderr)
        sys.exit(2)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
