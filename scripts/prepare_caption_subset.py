"""Rebuild the original caption subset from the official COCO Karpathy train JSON."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    selected = [row for row in rows if row["image"].startswith("val2014/")]
    if not selected:
        raise SystemExit("No val2014 images in the supplied caption training annotations")
    # Keep input order, original captions, fields, and original serialization.
    content = json.dumps(selected, ensure_ascii=False).encode("utf-8")
    target = args.root.resolve() / "data/coco/annotations/coco_stage1_existing_images.json"
    if target.exists() and target.read_bytes() != content:
        raise SystemExit("A different annotation file already exists; it was not overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(f"{len(selected)} captions / {len({r['image'] for r in selected})} images")
    print(f"SHA256: {hashlib.sha256(content).hexdigest()}")
    print(target)


if __name__ == "__main__":
    main()
