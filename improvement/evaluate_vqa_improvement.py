"""Evaluate an improvement checkpoint on fixed dev/test with official VQA scoring."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time


COMMIT = "8491f1d333cce678b1230cb940fb249253f3ea4d"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation split is empty or is not a list")
    ids = []
    for row in rows:
        ids.append(int(row["question_id"]))
        int(row["image_id"])
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("Question must be a nonempty string")
        if len(row["answers"]) != 10 or not all(isinstance(a, str) for a in row["answers"]):
            raise ValueError("Each evaluation question needs its 10 original answer votes")
        if row["answer_type"] not in ("yes/no", "number", "other"):
            raise ValueError("Original VQA answer_type is missing or invalid")
        if not isinstance(row["question_type"], str):
            raise ValueError("Original question_type is missing")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation question_id")


def official_documents(rows):
    validate_rows(rows)
    metadata = dict(info={"description": "Local image-disjoint VQA improvement experiment"},
                    task_type="Open-Ended", data_type="mscoco", data_subtype="val2014", license=[])
    questions = dict(metadata)
    questions["questions"] = [dict(question_id=int(r["question_id"]), image_id=int(r["image_id"]),
                                   question=r["question"]) for r in rows]
    annotations = dict(metadata)
    annotations["annotations"] = [dict(
        question_id=int(r["question_id"]), image_id=int(r["image_id"]),
        question_type=r["question_type"], answer_type=r["answer_type"],
        multiple_choice_answer=max(dict.fromkeys(r["answers"]), key=r["answers"].count),
        answers=[dict(answer=answer, answer_id=i + 1, answer_confidence="yes")
                 for i, answer in enumerate(r["answers"])]) for r in rows]
    return questions, annotations


def _load_source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pinned scoring code: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_predictions(rows, predictions, output, source):
    """Use the pinned official leave-one-annotator-out scorer, not min(n/3)."""
    expected = {int(row["question_id"]) for row in rows}
    predicted = [int(row["question_id"]) for row in predictions]
    if len(predicted) != len(expected) or set(predicted) != expected:
        raise ValueError("Predictions must contain each selected question_id exactly once")
    if any(not isinstance(row["answer"], str) for row in predictions):
        raise ValueError("Predicted answers must be strings")
    questions, annotations = official_documents(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in ("questions.json", "annotations.json", "predictions.json")}
    for name, value in (("questions.json", questions), ("annotations.json", annotations),
                        ("predictions.json", predictions)):
        paths[name].write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    scorer_folder = Path(source) / "lavis/common/vqa_tools"
    VQA = _load_source_module(scorer_folder / "vqa.py", "improvement_official_vqa").VQA
    VQAEval = _load_source_module(scorer_folder / "vqa_eval.py", "improvement_official_vqa_eval").VQAEval
    reference = VQA(str(paths["annotations.json"]), str(paths["questions.json"]))
    result = reference.loadRes(str(paths["predictions.json"]), str(paths["questions.json"]))
    scorer = VQAEval(reference, result, n=2)
    scorer.evaluate()
    metrics = {"agg_metrics": scorer.accuracy["overall"],
               **scorer.accuracy["perAnswerType"]}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Store image IDs as well as question IDs for image-clustered paired analysis.
    by_question = {int(row["question_id"]): row for row in predictions}
    detail = [dict(question_id=int(r["question_id"]), image_id=int(r["image_id"]),
                   answer_type=r["answer_type"], prediction=by_question[int(r["question_id"])]["answer"],
                   score_percent=scorer.evalQA[int(r["question_id"])]) for r in rows]
    (output / "per_question_scores.json").write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive; workers must be nonnegative")
    root = args.root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    data = root / "data/coco/improvement_v1"
    split_path = data / f"{args.split}.json"
    manifest_path = data / "split_manifest.json"
    for path in (checkpoint, split_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"缺少文件：{path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("outputs", {}).get(args.split, {})
    split_sha = sha256_file(split_path)
    if declared.get("sha256") != split_sha or declared.get("bytes") != split_path.stat().st_size:
        raise SystemExit("题集与固定 split_manifest 不一致，停止评估。")
    rows = json.loads(split_path.read_text(encoding="utf-8"))
    validate_rows(rows)
    images = root / "data/coco/images"
    for image in {row["image"] for row in rows}:
        path = (images / image).resolve()
        if not path.is_relative_to(images.resolve()) or not path.is_file():
            raise SystemExit(f"图片路径无效或缺少图片：{path}")
    print(f"固定 {args.split} 集：{len(rows)} 道题，{len({r['image_id'] for r in rows})} 张图片。", flush=True)
    print(f"模型检查点：{checkpoint}", flush=True)
    if args.prepare_only:
        print("数据和路径检查通过；尚未加载模型或验证 GPU。", flush=True)
        return
    if args.split == "test":
        print("执行显式最终 test 评估；请勿用 test 结果选择超参数或检查点。", flush=True)

    from question_gate import configure_runtime, load_improved_model
    paths = configure_runtime(root)
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from lavis.processors.blip_processors import BlipImageEvalProcessor, BlipQuestionProcessor
    if not torch.cuda.is_available():
        raise SystemExit("当前环境无法使用 CUDA。")
    model, checkpoint_meta = load_improved_model(root, checkpoint, device="cuda")
    if checkpoint_meta.get("run_config", {}).get("split_manifest_sha256") != sha256_file(manifest_path):
        raise SystemExit("模型训练时的划分清单与当前清单不同，停止评估，避免混用实验。")
    if args.split == "test" and checkpoint_meta.get("step") != checkpoint_meta.get("total_steps"):
        raise SystemExit("最终 test 仅用于完成计划训练的检查点；中途调试请使用 dev。")
    variant = checkpoint_meta["model_config"]["variant"]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = checkpoint.parent / f"eval_{args.split}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    visual = BlipImageEvalProcessor(image_size=224)
    question_processor = BlipQuestionProcessor()

    class EvalDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            with Image.open(images / row["image"]) as opened:
                pixels = visual(opened.convert("RGB"))
            return dict(image=pixels, question=question_processor(row["question"]),
                        question_id=int(row["question_id"]))

    loader = DataLoader(EvalDataset(), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    predictions = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            answers = model.answer_questions(batch["image"].to("cuda", non_blocking=True),
                                             batch["question"], num_beams=5,
                                             max_new_tokens=10, min_length=1)
            if len(answers) != len(batch["question_id"]):
                raise RuntimeError("模型回答数量与 batch 问题数不一致。")
            predictions.extend(dict(question_id=int(qid), answer=answer)
                               for qid, answer in zip(batch["question_id"], answers))
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(loader):
                print(f"Evaluation {variant}/{args.split}: {len(predictions)}/{len(rows)} questions", flush=True)
    inference_seconds = time.monotonic() - started
    metrics = score_predictions(rows, predictions, output, paths["source"])
    metadata = dict(variant=variant, split=args.split, question_count=len(rows),
                    checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                    split_sha256=split_sha, split_manifest_sha256=sha256_file(manifest_path),
                    checkpoint_metadata=checkpoint_meta, source_commit=COMMIT,
                    decoding=dict(num_beams=5, max_new_tokens=10, min_length=1),
                    batch_size=args.batch_size, inference_seconds=inference_seconds,
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
                    protocol="Custom supervised VQA experiment, not official VQAv2 train/test split",
                    selection_policy="Use dev only for model selection; test runs require explicit --split test")
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f"评估结果、逐题得分与元信息：{output}", flush=True)


if __name__ == "__main__":
    main()
