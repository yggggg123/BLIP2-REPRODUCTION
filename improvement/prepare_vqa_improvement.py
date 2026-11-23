"""Prepare reproducible, image-disjoint VQA splits from existing local COCO data.

No network download and no model loading. These are custom splits of VQAv2
validation images for a small supervised adaptation experiment, not the official
VQAv2 train/validation/test protocol.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import sys
import zipfile


VERSION = "image_disjoint_vqa_v1"
OUTPUT_FILES = {name: f"{name}.json" for name in ("train", "dev", "test")}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def image_key(row_or_name):
    name = row_or_name["image"] if isinstance(row_or_name, dict) else row_or_name
    if not isinstance(name, str):
        raise ValueError("image must be a string")
    name = name.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"Unsafe image path: {name}")
    return path.name


def question_id(row):
    value = row["question_id"]
    if isinstance(value, bool):
        raise ValueError("Boolean question_id is invalid")
    return int(value)


def index_questions(records, description):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{description} must be a nonempty list")
    indexed = {}
    for row in records:
        qid = question_id(row)
        if qid in indexed:
            raise ValueError(f"Duplicate question_id in {description}: {qid}")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Empty question in {description}: {qid}")
        image_key(row)
        indexed[qid] = row
    return indexed


def choose_splits(records, training_image_names, dev_records, train_count=10000,
                  test_count=1000, seed=42):
    """Select raw records deterministically, with no shared image across splits.

    Training samples at most one question per pretraining image. Dev question
    IDs are kept exactly; final-test questions are sampled from images unused
    by this experiment's caption pretraining and by the dev set.
    """
    if train_count < 1 or test_count < 1:
        raise ValueError("train_count and test_count must be positive")
    full = index_questions(records, "full VQA input")
    dev_input = index_questions(dev_records, "existing development set")
    training_names = {image_key(name) for name in training_image_names}
    if not training_names:
        raise ValueError("Caption training image set is empty")

    dev = []
    for qid, old in dev_input.items():
        if qid not in full:
            raise ValueError(f"Development question is missing from full VQA input: {qid}")
        current = full[qid]
        if image_key(old) != image_key(current) or old["question"] != current["question"]:
            raise ValueError(f"Development question/image differs from original VQA input: {qid}")
        dev.append(current)
    dev.sort(key=question_id)
    dev_names = {image_key(row) for row in dev}
    overlap = training_names & dev_names
    if overlap:
        raise ValueError(f"Development images overlap caption training images: {sorted(overlap)[:3]}")

    train_by_image = defaultdict(list)
    candidates = []
    for qid in sorted(full):
        row = full[qid]
        name = image_key(row)
        if name in training_names:
            train_by_image[name].append(row)
        elif name not in dev_names:
            candidates.append(row)
    if train_count > len(train_by_image):
        raise ValueError(f"Need {train_count} unique training images, only {len(train_by_image)} available")
    if test_count > len(candidates):
        raise ValueError(f"Need {test_count} final-test questions, only {len(candidates)} eligible")

    train_rng = random.Random(seed)
    names = train_rng.sample(sorted(train_by_image), train_count)
    train = [train_rng.choice(train_by_image[name]) for name in names]
    train.sort(key=question_id)
    test = random.Random(seed + 1).sample(candidates, test_count)
    test.sort(key=question_id)
    splits = {"train": train, "dev": dev, "test": test}
    validate_disjoint(splits)
    if len({image_key(row) for row in train}) != len(train):
        raise ValueError("Training selection contains multiple questions from one image")
    return splits


def validate_disjoint(splits):
    images = {}
    questions = {}
    for name, rows in splits.items():
        indexed = index_questions(rows, f"{name} split")
        images[name] = {image_key(row) for row in rows}
        questions[name] = set(indexed)
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if images[left] & images[right]:
                raise ValueError(f"Images overlap between {left} and {right}")
            if questions[left] & questions[right]:
                raise ValueError(f"Question IDs overlap between {left} and {right}")


def load_json_source(root, relative_path, zip_member=None):
    """Load a local source, optionally reading a fixed entry from a local ZIP."""
    path = root / relative_path
    if path.is_file():
        content = path.read_bytes()
        location = relative_path
    else:
        if not zip_member:
            raise FileNotFoundError(f"Missing source: {path}")
        content = None
        for bundle in (root / "data/coco/vqav2_annotations.zip", root / "vqav2_annotations.zip"):
            if bundle.is_file():
                with zipfile.ZipFile(bundle) as archive:
                    if zip_member in archive.namelist():
                        content = archive.read(zip_member)
                        location = bundle.relative_to(root).as_posix() + "!" + zip_member
                        break
        if content is None:
            raise FileNotFoundError(
                f"Missing source: {path}; no local annotation ZIP contains {zip_member}. "
                "Use the original local annotation files; this helper does not download data."
            )
    fingerprint = {"location": location, "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}
    return json.loads(content.decode("utf-8-sig")), fingerprint


def enrich_splits(splits, raw_annotation_document):
    """Use original VQA ten-vote answers and official answer/question types."""
    wanted = {question_id(row) for rows in splits.values() for row in rows}
    annotations = raw_annotation_document.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Raw VQA document is missing its annotations list")
    raw = {}
    for ann in annotations:
        qid = question_id(ann)
        if qid in wanted:
            if qid in raw:
                raise ValueError(f"Duplicate selected question ID in raw VQA annotation: {qid}")
            raw[qid] = ann
    if set(raw) != wanted:
        raise ValueError(f"Raw annotation is missing {len(wanted - set(raw))} selected questions")

    enriched = {}
    for name, rows in splits.items():
        enriched[name] = []
        for row in rows:
            qid = question_id(row)
            ann = raw[qid]
            votes = ann.get("answers")
            if not isinstance(votes, list) or len(votes) != 10:
                raise ValueError(f"Question {qid} does not contain exactly ten human answers")
            answers = [vote["answer"] for vote in votes]
            if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
                raise ValueError(f"Question {qid} contains an invalid human answer")
            relative_image = row["image"].replace("\\", "/")
            path = PurePosixPath(relative_image)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "val2014":
                raise ValueError(f"Unexpected VQAv2 validation image path: {relative_image}")
            match = re.fullmatch(r"COCO_val2014_(\d+)\.jpg", path.name)
            if not match or int(match.group(1)) != int(ann["image_id"]):
                raise ValueError(f"Raw annotation image_id disagrees with image file: {qid}")
            answer_type = ann.get("answer_type")
            question_type = ann.get("question_type")
            if not isinstance(answer_type, str) or not isinstance(question_type, str):
                raise ValueError(f"Raw annotation lacks official VQA types: {qid}")
            enriched[name].append({
                "image": relative_image, "image_id": int(ann["image_id"]),
                "question_id": qid, "question": row["question"], "answers": answers,
                "answer_type": answer_type, "question_type": question_type,
            })
    validate_disjoint(enriched)
    return enriched


def check_images(root, splits):
    image_root = (root / "data/coco/images").resolve()
    names = sorted({row["image"] for rows in splits.values() for row in rows})
    missing = []
    for name in names:
        path = (image_root / name).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError(f"Image path escapes image root: {name}")
        if not path.is_file():
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images, examples: {missing[:5]}")


def source_hashes(fingerprints):
    return {name: {key: info[key] for key in ("bytes", "sha256")}
            for name, info in fingerprints.items()}


def verify_existing(output, parameters, fingerprints):
    manifest_path = output / "split_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("parameters") != parameters:
        raise ValueError("Existing split parameters/version differ; use a new experiment or explicit --force")
    if source_hashes(manifest.get("source_fingerprints", {})) != source_hashes(fingerprints):
        raise ValueError("Original source hashes changed; refusing to silently replace existing splits")
    splits = {}
    for name, filename in OUTPUT_FILES.items():
        entry = manifest.get("outputs", {}).get(name, {})
        path = output / filename
        if entry.get("file") != filename or not path.is_file():
            raise ValueError(f"Existing split is incomplete: {path}")
        if path.stat().st_size != entry.get("bytes") or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"Existing split hash/size changed: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != entry.get("questions") or len({image_key(row) for row in rows}) != entry.get("images"):
            raise ValueError(f"Existing split counts disagree with manifest: {path}")
        splits[name] = rows
    validate_disjoint(splits)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Explicitly replace only this helper's four split files.")
    parser.add_argument("--skip-image-check", action="store_true", help="Prepare annotations only; images remain unverified.")
    args = parser.parse_args()
    if args.train_count < 1 or args.test_count < 1:
        parser.error("train-count and test-count must be positive")
    root = args.root.expanduser().resolve()
    output = root / "data/coco/improvement_v1"
    if not output.resolve().is_relative_to(root):
        raise SystemExit("Output folder resolves outside the project root")
    parameters = {"train_count": args.train_count, "test_count": args.test_count, "seed": args.seed,
                  "train_rule": "one_question_per_caption_training_image", "dev_rule": "fixed_previous_1000_questions",
                  "test_rule": "exclude_all_caption_training_and_dev_images"}
    try:
        print("Reading existing local annotations; no network download...", flush=True)
        records, full_fp = load_json_source(root, "data/coco/annotations/vqa_val_eval.json", "vqa_val_eval.json")
        caption_rows, caption_fp = load_json_source(root, "data/coco/annotations/coco_stage1_existing_images.json")
        dev_rows, dev_fp = load_json_source(root, "data/coco/annotations/holdout_n1000_seed42/vqa_val_eval.json")
        if not isinstance(dev_rows, list) or len(dev_rows) != 1000:
            raise ValueError("Existing development input must be the previously evaluated 1000 questions")
        fingerprints = {"full_vqa_input": full_fp, "caption_training": caption_fp, "development": dev_fp}
        # The raw source hash is checked as well, even if previously generated splits exist.
        raw, raw_fp = load_json_source(root, "data/coco/annotations/v2_mscoco_val2014_annotations.json",
                                       "v2_mscoco_val2014_annotations.json")
        fingerprints["raw_vqa_answers"] = raw_fp
        if not args.force:
            existing = verify_existing(output, parameters, fingerprints)
            if existing is not None:
                if not args.skip_image_check:
                    check_images(root, existing)
                print(f"Existing splits verified and reused: {output}", flush=True)
                for name, rows in existing.items():
                    print(f"{name}: {len(rows)} questions / {len({image_key(row) for row in rows})} images", flush=True)
                return
            if any((output / name).exists() for name in OUTPUT_FILES.values()):
                raise ValueError("Split files exist without a valid manifest; refusing to overwrite. Use explicit --force if intended")
        splits = choose_splits(records, [row["image"] for row in caption_rows], dev_rows,
                               args.train_count, args.test_count, args.seed)
        del records, caption_rows, dev_rows
        splits = enrich_splits(splits, raw)
        del raw
        if not args.skip_image_check:
            check_images(root, splits)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {}
        for name, rows in splits.items():
            path = output / OUTPUT_FILES[name]
            content = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            temporary = path.with_suffix(".json.part")
            temporary.write_bytes(content)
            temporary.replace(path)
            metadata[name] = {"file": path.name, "bytes": len(content),
                              "sha256": hashlib.sha256(content).hexdigest(),
                              "questions": len(rows), "images": len({image_key(row) for row in rows}),
                              "question_ids_sha256": hashlib.sha256(json.dumps(sorted(question_id(row) for row in rows)).encode()).hexdigest()}
        manifest = {"version": VERSION,
                    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    "protocol": "Custom supervised VQA splits from locally available VQAv2 validation images; not official benchmark splits",
                    "parameters": parameters, "source_fingerprints": fingerprints, "outputs": metadata,
                    "images_checked": not args.skip_image_check,
                    "answers": "Ten original human-answer votes per question; preserve frequency when sampling training targets",
                    "pretraining_scope": "Disjointness refers to this local caption pretraining, not all data used by public EVA/T5 checkpoints"}
        target = output / "split_manifest.json"
        temporary = target.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        print(f"Prepared image-disjoint splits: {output}", flush=True)
        for name, info in metadata.items():
            print(f"{name}: {info['questions']} questions / {info['images']} images", flush=True)
        if args.skip_image_check:
            print("Image paths were NOT checked (--skip-image-check).", flush=True)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Split preparation stopped: {error}") from error


if __name__ == "__main__":
    main()
