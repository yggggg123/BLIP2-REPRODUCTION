"""Protocol tests for image-disjoint small-scale VQA improvement splits."""

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
import zipfile


MODULE = Path(__file__).resolve().parents[1] / "prepare_vqa_improvement.py"
spec = importlib.util.spec_from_file_location("prepare_splits", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def image(index):
    return f"val2014/COCO_val2014_{index:012d}.jpg"


def fixture_records():
    return [{"question_id": i * 10 + q, "question": f"Question {i} {q}?", "image": image(i)}
            for i in range(1, 21) for q in range(3)]


def raw_annotations(records):
    return {"annotations": [{"question_id": row["question_id"],
                              "image_id": int(Path(row["image"]).stem.rsplit("_", 1)[1]),
                              "answers": [{"answer": "yes", "answer_id": i} for i in range(10)],
                              "answer_type": "yes/no", "question_type": "is this"}
                             for row in records]}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.records = fixture_records()
        self.training = [image(i) for i in range(1, 11)]
        self.dev = [row for row in self.records if row["question_id"] in {110, 111, 120}]

    def test_image_disjoint_and_one_train_question_per_image(self):
        splits = m.choose_splits(self.records, self.training, self.dev, 7, 10, 42)
        self.assertEqual(len(splits["train"]), 7)
        self.assertEqual(len({row["image"] for row in splits["train"]}), 7)
        self.assertEqual({row["question_id"] for row in splits["dev"]}, {110, 111, 120})
        self.assertTrue(all(row["image"] not in {image(11), image(12)} for row in splits["test"]))
        self.assertTrue(all(row["image"] not in self.training for row in splits["test"]))
        m.validate_disjoint(splits)

    def test_input_order_does_not_change_selection(self):
        expected = m.choose_splits(self.records, self.training, self.dev, 7, 10, 42)
        changed = self.records.copy()
        random.Random(55).shuffle(changed)
        self.assertEqual(expected, m.choose_splits(changed, list(reversed(self.training)),
                                                 list(reversed(self.dev)), 7, 10, 42))

    def test_invalid_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            m.choose_splits(self.records, self.training, self.dev, 11, 10)
        with self.assertRaises(ValueError):
            m.choose_splits(self.records + [self.records[0]], self.training, self.dev, 7, 10)
        with self.assertRaises(ValueError):
            m.choose_splits(self.records, self.training, self.dev + [self.records[0]], 7, 10)
        bad_dev = [dict(self.dev[0], question="changed question")]
        with self.assertRaises(ValueError):
            m.choose_splits(self.records, self.training, bad_dev, 7, 10)
        with self.assertRaises(ValueError):
            m.image_key("../outside.jpg")

    def test_original_ten_votes_and_vqa_types_are_preserved(self):
        splits = m.choose_splits(self.records, self.training, self.dev, 7, 10)
        raw = raw_annotations(self.records)
        raw["annotations"][0]["answers"][0]["answer"] = "no"
        enriched = m.enrich_splits(splits, raw)
        for rows in enriched.values():
            for row in rows:
                self.assertEqual(len(row["answers"]), 10)
                self.assertEqual(row["answer_type"], "yes/no")
                self.assertEqual(row["question_type"], "is this")
                self.assertIsInstance(row["image_id"], int)
        selected_id = enriched["train"][0]["question_id"]
        entry = next(row for row in raw["annotations"] if row["question_id"] == selected_id)
        entry["answers"].pop()
        with self.assertRaises(ValueError):
            m.enrich_splits(splits, raw)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[2]
        self.root = Path(tempfile.mkdtemp(prefix="split_test_", dir=self.workspace)).resolve()
        self.assertTrue(self.root.is_relative_to(self.workspace))

    def tearDown(self):
        resolved = self.root.resolve()
        self.assertTrue(resolved.is_relative_to(self.workspace))
        self.assertTrue(resolved.name.startswith("split_test_"))
        shutil.rmtree(resolved)

    def test_zip_fallback_and_content_fingerprint(self):
        payload = b'[{"question_id":1}]'
        with zipfile.ZipFile(self.root / "vqav2_annotations.zip", "w") as archive:
            archive.writestr("vqa_val_eval.json", payload)
        rows, fingerprint = m.load_json_source(self.root, "data/coco/annotations/vqa_val_eval.json", "vqa_val_eval.json")
        self.assertEqual(rows, [{"question_id": 1}])
        self.assertEqual(fingerprint["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(fingerprint["bytes"], len(payload))
        self.assertFalse((self.root / "data").exists())

    def test_manifest_reuse_and_tamper_rejection(self):
        records = fixture_records()
        splits = m.choose_splits(records, [image(i) for i in range(1, 11)],
                                 [row for row in records if row["question_id"] in {110, 111, 120}], 7, 10)
        parameters = {"seed": 42}
        sources = {"input": {"location": "original.json", "bytes": 2, "sha256": "abc"}}
        output = self.root / "output"
        output.mkdir()
        entries = {}
        for name, rows in splits.items():
            path = output / f"{name}.json"
            content = json.dumps(rows).encode()
            path.write_bytes(content)
            entries[name] = {"file": path.name, "bytes": len(content),
                             "sha256": hashlib.sha256(content).hexdigest(), "questions": len(rows),
                             "images": len({row["image"] for row in rows})}
        manifest = {"version": m.VERSION, "parameters": parameters, "source_fingerprints": sources,
                    "outputs": entries}
        (output / "split_manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(m.verify_existing(output, parameters, sources), splits)
        with self.assertRaises(ValueError):
            m.verify_existing(output, {"seed": 43}, sources)
        changed = {"input": {"location": "original.json", "bytes": 2, "sha256": "def"}}
        with self.assertRaises(ValueError):
            m.verify_existing(output, parameters, changed)
        (output / "test.json").write_text("[]")
        with self.assertRaises(ValueError):
            m.verify_existing(output, parameters, sources)


if __name__ == "__main__":
    unittest.main()
