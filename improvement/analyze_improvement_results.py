"""Compare saved VQA predictions with paired image-cluster bootstrap. CPU only.

Resamples images, preserving all questions for each selected image. Intervals
describe uncertainty under image sampling, not training-seed variability.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random


VARIANTS = ("plain", "constant", "question")


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_interval(reference, candidate, resamples=2000, seed=2026):
    def index(rows):
        result = {int(row["question_id"]): row for row in rows}
        if not result or len(result) != len(rows):
            raise ValueError("Empty or duplicate question IDs")
        return result
    left, right = index(reference), index(candidate)
    if set(left) != set(right):
        raise ValueError("The models were not scored on identical questions")
    groups = defaultdict(lambda: [0.0, 0])
    for qid, row in left.items():
        other = right[qid]
        if int(row["image_id"]) != int(other["image_id"]):
            raise ValueError("Image IDs disagree for a matched question")
        a, b = float(row["score_percent"]), float(other["score_percent"])
        if not 0 <= a <= 100 or not 0 <= b <= 100:
            raise ValueError("Invalid per-question percentage")
        group = groups[int(row["image_id"])]
        group[0] += b - a
        group[1] += 1
    clusters = [groups[key] for key in sorted(groups)]
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        numerator = denominator = 0.0
        for _ in clusters:
            difference, count = clusters[rng.randrange(len(clusters))]
            numerator += difference
            denominator += count
        differences.append(numerator / denominator)
    return {"difference_pp": sum(g[0] for g in clusters) / len(left),
            "ci95_low_pp": percentile(differences, 0.025),
            "ci95_high_pp": percentile(differences, 0.975),
            "questions": len(left), "images": len(clusters), "resamples": resamples}


def read_run(base, variant, split, seed, steps):
    folder = base / f"{variant}_seed{seed}_steps{steps}"
    candidates = list(folder.glob(f"eval_{split}_*/metrics.json"))
    if not candidates:
        raise ValueError(f"Missing saved {variant}/{split} evaluation under {folder}")
    path = max(candidates, key=lambda item: item.stat().st_mtime).parent
    data = {name: json.loads((path / filename).read_text(encoding="utf-8")) for name, filename in
            (("metrics", "metrics.json"), ("metadata", "metadata.json"),
             ("questions", "per_question_scores.json"))}
    if data["metadata"]["variant"] != variant or data["metadata"]["split"] != split:
        raise ValueError("Evaluation metadata disagrees with its directory")
    mean = sum(float(r["score_percent"]) for r in data["questions"]) / len(data["questions"])
    if abs(mean - data["metrics"]["agg_metrics"]) > 0.02:
        raise ValueError("Saved per-question scores disagree with the aggregate metric")
    data["path"] = str(path)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--seed", type=int, default=42, help="Training seed of saved models")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("Use at least 100 bootstrap resamples")
    base = args.root.expanduser().resolve() / "output/improvement"
    runs = {v: read_run(base, v, args.split, args.seed, args.steps) for v in VARIANTS}
    first = runs["plain"]["metadata"]
    keys = ("seed", "total_steps", "batch_size", "base_lr", "train_sha256",
            "split_manifest_sha256", "sample_plan_sha256")
    for v in VARIANTS:
        meta = runs[v]["metadata"]
        if meta["split_sha256"] != first["split_sha256"] or meta["decoding"] != first["decoding"]:
            raise ValueError("Models use different evaluation questions or decoding")
        checkpoint = meta["checkpoint_metadata"]
        baseline = first["checkpoint_metadata"]
        if checkpoint["init_checkpoint_sha256"] != baseline["init_checkpoint_sha256"]:
            raise ValueError("Models did not start from the same checkpoint file")
        if any(checkpoint["run_config"][key] != baseline["run_config"][key] for key in keys):
            raise ValueError("Training data, plan, seed or common budget differs")
    comparisons = {}
    for a, b in (("plain", "constant"), ("plain", "question"), ("constant", "question")):
        comparisons[f"{b}-{a}"] = paired_interval(runs[a]["questions"], runs[b]["questions"], args.resamples)
    lines = [f"# BLIP-2 改进实验：{args.split} 集配对分析", "",
             "| 模型 | 总体 | 数字类 | 其他 | 是／否 |", "|---|---:|---:|---:|---:|"]
    for v in VARIANTS:
        score = runs[v]["metrics"]
        lines.append(f"| {v} | {score['agg_metrics']:.2f} | {score['number']:.2f} | "
                     f"{score['other']:.2f} | {score['yes/no']:.2f} |")
    lines += ["", "| 比较（前者减后者，百分点） | 差值 | 图片配对 bootstrap 95% 区间 |",
              "|---|---:|---:|"]
    for name, value in comparisons.items():
        lines.append(f"| {name} | {value['difference_pp']:+.3f} | "
                     f"[{value['ci95_low_pp']:+.3f}, {value['ci95_high_pp']:+.3f}] |")
    lines += ["", "区间若包含 0，本次图片抽样下的结果不足以支持明确方向的收益。"
              "即使区间不包含 0，也不能替代多训练随机种子实验，或证明跨数据集泛化。",
              "", "这是一组单训练种子、同预算、小规模监督 VQA 实验。"
              "已检查初始化、训练数据／样本计划、公共学习率、步数、评估题集与解码一致。"
              "该统计不训练、不再次调用模型，不改变最终测试集，也不根据测试分数调整参数。"]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = base / f"analysis_{args.split}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    report = "\n".join(lines) + "\n"
    (output / "comparison.md").write_text(report, encoding="utf-8")
    payload = {"split": args.split, "training_seed": args.seed, "comparisons": comparisons,
               "metrics": {v: runs[v]["metrics"] for v in VARIANTS},
               "source_directories": {v: runs[v]["path"] for v in VARIANTS}}
    (output / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report, flush=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
