"""Archive small-scale BLIP-2 evidence without duplicating large weights on disk.

Run this file in the server project directory after activating its .venv.
Only Python's standard library is required; installed Torch is used optionally
to read the user's own checkpoint metadata on CPU. No training or GPU allocation.
"""

import argparse
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


COMMIT = "8491f1d333cce678b1230cb940fb249253f3ea4d"
ALLOWED = {".py", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".log", ".csv", ".md"}
EXPECTED = {
    "stage2": {"agg_metrics": 41.90, "yes/no": 70.05, "number": 14.85, "other": 26.59},
    "official": {"agg_metrics": 65.48, "yes/no": 85.27, "number": 45.88, "other": 54.88},
}


def sha256_file(path):
    digest = hashlib.sha256()
    size = path.stat().st_size
    read = 0
    last = time.monotonic()
    with path.open("rb") as stream:
        while True:
            block = stream.read(4 * 1024**2)
            if not block:
                break
            digest.update(block)
            read += len(block)
            if time.monotonic() - last > 5:
                print(f"  SHA256 {path.name}: {read / 1024**3:.2f}/{size / 1024**3:.2f} GiB", flush=True)
                last = time.monotonic()
    return digest.hexdigest()


def newest(paths, description):
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(f"找不到{description}，尚未生成备份。请检查服务器文件路径。")
    return max(paths, key=lambda p: p.stat().st_mtime)


def last_metrics(path):
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "agg_metrics" in row:
            return {key: float(row[key]) for key in ("agg_metrics", "yes/no", "number", "other")}
    raise SystemExit(f"找不到完整 VQA 指标：{path}")


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45,
                                encoding="utf-8", errors="replace")
        return result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
    except (OSError, subprocess.SubprocessError) as error:
        return f"Unable to run {command[0]}: {error}\n"


def checkpoint_metadata(path, warnings):
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        result = {key: payload[key] for key in
                  ("kind", "step", "total_steps", "batch_size", "iters", "epoch", "stage1",
                   "data_scope", "prefix_split") if key in payload}
        state = payload.get("model", {})
        result["model_tensor_count"] = len(state)
        result["has_optimizer"] = "optimizer" in payload
        del payload, state
        gc.collect()
        return result
    except Exception as error:
        warnings.append(f"无法读取检查点元信息（原文件仍会登记校验值）：{path.name}: {error}")
        return {}


def log_summary(path):
    if not path.is_file():
        return {}
    by_step = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            by_step[int(row["step"])] = row
        except (ValueError, TypeError, KeyError):
            continue
    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        return {}
    return {"recorded_points": len(rows), "first_record": rows[0], "last_record": rows[-1],
            "note": "日志为间隔采样；随机 caption 前后缀使 loss 波动，不能据此单独判断收敛。"}


def compact_payload(payload, role, source_path):
    """Keep tensor values/dtypes, removing frozen backbones and training state."""
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("检查点中缺少 model 参数字典")
    if role == "stage2" and payload.get("kind") != "blip2_local_stage2_v1":
        raise ValueError("第二阶段权重不是本项目训练脚本生成的格式")
    selected = {name: tensor for name, tensor in state.items()
                if not name.startswith(("visual_encoder.", "t5_model."))}
    required = {"query_tokens", "ln_vision.weight", "ln_vision.bias"}
    if role == "stage2":
        required |= {"t5_proj.weight", "t5_proj.bias"}
    if not required.issubset(selected) or not any(name.startswith("Qformer.") for name in selected):
        raise ValueError("检查点缺少 Q-Former、query、视觉 LayerNorm 或 T5 投影参数")
    # Clone each retained tensor to avoid serializing a larger backing storage.
    # No FP16 conversion or quantization: values and dtypes remain unchanged.
    selected = {name: tensor.detach().cpu().clone() for name, tensor in selected.items()}
    result = {key: payload[key] for key in
              ("step", "total_steps", "batch_size", "iters", "epoch", "stage1", "data_scope", "prefix_split")
              if key in payload}
    result.update(model=selected, kind=("blip2_local_stage2_v1" if role == "stage2"
                                       else "blip2_local_stage1_weights_v1"),
                  inference_only=True, source_checkpoint=str(source_path), source_commit=COMMIT,
                  export_note="Exact tensor values/dtypes; frozen backbones, optimizer, scaler and RNG state omitted.")
    return result


def export_compact(root, source_paths):
    import torch
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = root / "compact_weights" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    exported = {}
    for role, path in source_paths.items():
        print(f"CPU 精简导出 {role}：读取 {path.name}……", flush=True)
        original = torch.load(path, map_location="cpu", weights_only=False)
        payload = compact_payload(original, role, path)
        del original
        gc.collect()
        estimated = sum(t.numel() * t.element_size() for t in payload["model"].values())
        if shutil.disk_usage(root).free < estimated + 64 * 1024**2:
            raise SystemExit(f"导出 {role} 约需 {(estimated + 64 * 1024**2) / 1024**3:.2f} GiB 空闲空间。原文件保持不变。")
        target = folder / f"{role}_model_only.pth"
        temporary = target.with_suffix(".pth.part")
        torch.save(payload, temporary)
        temporary.replace(target)
        exported[role] = target
        print(f"{role}：{path.stat().st_size / 1024**3:.2f} → "
              f"{target.stat().st_size / 1024**3:.2f} GiB；原权重保留。", flush=True)
        del payload
        gc.collect()
    return exported


def make_report(data):
    scores = data["scores"]
    own, official = scores["stage2"], scores["official"]
    labels = [("总体（VQA 软准确率）", "agg_metrics"), ("是／否", "yes/no"),
              ("数字类", "number"), ("其他", "other")]
    table = "\n".join(f"| {label} | {own[key]:.2f} | {official[key]:.2f} | "
                      f"{official[key] - own[key]:.2f} |" for label, key in labels)
    meta = data.get("checkpoint_metadata", {})
    stage2 = meta.get("stage2", {})
    log = data.get("training_log_summary", {})
    actual = (f"服务器检查点记录：第二阶段完成 {stage2['step']}/{stage2.get('total_steps', '未记录')} 步，"
              f"batch size 为 {stage2.get('batch_size', '未记录')}。"
              if "step" in stage2 else
              "阶段一、二各完成 5000 步；最终步数与实际 batch size 以归档的服务器配置和检查点元信息为准。")
    log_text = "尚未读取服务器训练日志，本文不报告训练 loss 数值、训练时长或收敛结论。"
    if log:
        first, last = log["first_record"], log["last_record"]
        log_text = (f"归档日志首条记录为第 {first['step']} 步，loss={first['loss']:.4f}；"
                    f"末条记录为第 {last['step']} 步，loss={last['loss']:.4f}。"
                    "这些是间隔采样的训练损失，不等同于验证集损失，也不足以单独证明收敛。")
    warnings = data.get("warnings", [])
    warning_text = ("\n\n归档时待核对事项：\n\n" + "\n".join(f"- {item}" for item in warnings)) if warnings else ""
    question_text = str(data.get("question_count", 1000))
    training = data.get("training_data", {})
    images = training.get("images", 30504)
    pairs = training.get("caption_pairs", 152634)
    compact_note = ("\n\n本次下载的是精简模型权重：保留模型参数的原始数值和精度，"
                    "省略冻结骨干、优化器、scaler 和随机数状态。它可用于重新加载模型或作为新训练的初始化，"
                    "不能用于恢复原优化器状态并精确续训；完整检查点仍保留在服务器。"
                    if data.get("export_mode") == "model_only" else "")
    return f"""# BLIP-2 小规模两阶段复现实验报告


记录日期：{data.get('date', '2026-11-22')}　范围：单卡、小数据集、两阶段流程复现。

## 摘要

本实验在单张 NVIDIA A800 80GB 上，对 BLIP-2 进行小规模两阶段复现。第一阶段使用图文对训练 Q-Former，第二阶段接入冻结的 Flan-T5-XL，训练视觉信息到语言模型输入空间的连接。两阶段各执行 5000 步，并在排除本次训练图片后选取的 {question_text} 道 VQAv2 验证问题上评估。自训练模型的 VQA 软准确率为 {own['agg_metrics']:.2f}%，官方模型在相同题集上为 {official['agg_metrics']:.2f}%。实验完成了训练、权重保存、推理和定量评估流程；目标为在有限资源下实现核心流程，不以匹配论文原始训练规模或精度为验收条件。

## 1. 实验目标与方法

BLIP-2 通过 Q-Former 连接预训练视觉编码器与大语言模型。图像先由 EVA ViT-g 编码，32 个可学习 query 通过 Q-Former 提取视觉信息，再经线性层投影为 Flan-T5-XL 可接收的视觉软提示。视觉编码器和语言模型的骨干参数保持冻结。

第一阶段采用图文对比学习（ITC）、图文匹配（ITM）和图像条件文本生成（ITG），优化 Q-Former、query、视觉 LayerNorm 和相应任务头。第二阶段继承第一阶段保留下来的参数，初始化新的 T5 投影层；将 caption 随机按词边界分成前缀和后缀，用“视觉表示＋前缀”预测后缀。随机按词边界切分是本地脚本的实现选择，不能声称与原作者未公开的所有数据处理细节完全一致。

## 2. 环境与数据

| 项目 | 本实验设置 |
|---|---|
| GPU | NVIDIA A800-SXM4-80GB，1 张 |
| CPU／主机内存 | 18 核／120 GB（实例配置） |
| 数据盘 | 50 GB |
| Python／PyTorch／torchvision | 3.10.22／2.3.0+cu118／0.18.0+cu118（先前环境检查；归档环境清单记录最终版本） |
| Transformers | 安装约束为 4.28.1；最终版本见环境清单 |
| LAVIS | 固定提交 `{COMMIT}` |
| 视觉编码器／语言模型 | EVA ViT-g／Flan-T5-XL |
| 图像尺寸／query 数量 | 224×224／32 |
| 本地训练数据 | COCO Karpathy train 中可复用现有 val2014 图片的子集 |
| 图片／caption 数量 | {images}／{pairs} |

由于数据盘受限，本实验复用已下载的图片。图片位于 `val2014` 文件夹并不等于其 caption 属于 Karpathy validation；训练标注经过 Karpathy train 规则筛选。后续 VQAv2 评估又按图片排除了本次训练标注涉及的全部图片。

## 3. 训练设置

| 项目 | 第一阶段 | 第二阶段 |
|---|---|---|
| 训练步数 | 5000 | 5000 |
| batch size | 32（此前执行命令） | 16（此前执行命令） |
| 目标 | ITC＋ITM＋ITG | Prefix language modeling |
| 主要训练部分 | Q-Former、query、视觉 LayerNorm、任务头 | Q-Former、query、视觉 LayerNorm、T5 投影层 |
| 冻结骨干 | EVA | EVA、Flan-T5-XL |
| 优化器 | AdamW | AdamW |
| 峰值学习率 | 1e-4 | 1e-4 |
| 最低学习率配置 | 1e-5 | 5e-5 |
| 权重衰减／beta | 0.05／(0.9, 0.98) | 0.05／(0.9, 0.98) |
| 随机种子 | 42 | 42 |

{actual}

表中的学习率为脚本配置值。阶段一采用固定版本的 LAVIS 迭代 runner，阶段二采用本地脚本的逐步 warmup＋余弦调度；不能将配置名或最低学习率字段视为实际逐步学习率轨迹的证明。实际配置、日志与代码随结果一起保存。

若使用上述 batch size，阶段一和第二阶段分别处理约 160000 和 80000 对图文，约相当于现有 caption 条目数的 1.05 倍和 0.52 倍。这是处理量估算，不代表新增的唯一图片数量。

{log_text}

## 4. 评估协议

评估使用 VQAv2 validation 标注。首先排除本次训练集涉及的图片，再以 seed=42 从剩余问题中抽取 {question_text} 道。自训练模型和官方模型采用相同的 question_id 集合、图像预处理、提示词和解码配置。

- 提示词：`Question: {{}} Short answer:`。
- 生成参数：5-beam，max_len=10，min_len=1；其余沿用固定版本实现。
- 评估 batch size：此前命令为 64，实际配置随归档保存。
- 评分：沿用 LAVIS 内的 VQA 评分代码，报告百分制软准确率。
- 分数来源：{data.get('score_source', '已有结果截图；尚未在本地直接读取服务器文件')}。

保存评估输入、参考答案、预测答案和配置，便于复核题集和分数。总体分数不是三个答案类别得分的等权平均。

## 5. 实验结果

| 指标（%） | 自训练模型 | 官方模型 | 官方高出的百分点 |
|---|---:|---:|---:|
{table}

在该子集上，自训练模型总体低于官方模型 {official['agg_metrics'] - own['agg_metrics']:.2f} 个百分点。是／否类题目的得分较高，数字类题目的得分较低。这里的“数字类”采用 VQA 的 number 分类，不全部等同于计数任务。

## 6. 分析与边界

本实验已经获得可加载和可评估的两阶段训练权重，满足小规模流程复现的目标。与官方模型相比，本地训练数据量、训练步数和预训练经历均不同，这些差异可能影响结果；现有对照不能将差距归因于某一个单独因素。

当前没有“随机连接层”“仅第一阶段”等消融，也没有多随机种子重复实验。因此不能仅凭 41.90 分声称各阶段分别带来了多少提升，不能判断跨种子的稳定性，也不将 loss 下降等同于泛化能力提升。

图片排除规则仅保证题集未使用本次本地训练标注中的图片，不保证官方模型在其原始预训练中未见过这些图片。官方模型在本子集的 65.48 分和论文全量验证集的约 63.1 分属于不同评估集合，不直接比较。

## 7. 结论

本实验在单卡和有限磁盘条件下完成了 BLIP-2 两阶段训练、权重推理和同题集评估。自训练模型在定义好的 {question_text} 题子集上取得 {own['agg_metrics']:.2f}% 的 VQA 软准确率，并与官方模型完成对照。实验按小规模复现目标收尾，不继续扩大训练或评估规模。

## 8. 保存与复查

保存内容包括两阶段最终检查点、训练和评估脚本、固定源码及本地修改、配置、日志、评估输入与参考答案、预测结果、最终环境清单和 SHA256 校验清单。检查点与小文件证据包分开下载，避免在服务器复制一份大型权重。

检查点不包含恢复整个模型所需的全部独立基础权重。以后推理或续训时仍需要原来的 EVA、BERT、Flan-T5-XL 文件；基础权重可以另存，或根据清单重新取得。此处的备份不是整个 AutoDL 容器镜像。
{compact_note}

## 参考资料

1. [BLIP-2 原论文](https://proceedings.mlr.press/v202/li23q/li23q.pdf)。
2. [固定版本 LAVIS 源码](https://github.com/salesforce/LAVIS/tree/{COMMIT})。
3. [固定版本 VQAv2 评估配置](https://github.com/salesforce/LAVIS/blob/{COMMIT}/lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml)。
4. [VQAv2 数据集](https://visualqa.org/download.html)。
{warning_text}
"""


VERIFY_SCRIPT = '''"""Verify evidence files and separately downloaded checkpoints; no extra packages."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
failed = []
for entry in manifest["files"]:
    path = (root / entry["local_path"]).resolve()
    if not path.is_relative_to(root):
        raise SystemExit("Unexpected manifest path")
    if not path.is_file():
        print("MISSING:", entry["local_path"])
        failed.append(entry["local_path"])
        continue
    if path.stat().st_size != entry["size"]:
        print("SIZE MISMATCH:", entry["local_path"])
        failed.append(entry["local_path"])
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    good = digest.hexdigest() == entry["sha256"]
    print("OK:" if good else "HASH MISMATCH:", entry["local_path"])
    if not good:
        failed.append(entry["local_path"])
if failed:
    raise SystemExit(f"Verification failed for {len(failed)} files; keep the server originals.")
print("ALL FILES VERIFIED: evidence and both checkpoints are complete.")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stage1", type=Path, help="Override the automatically selected stage-1 checkpoint")
    parser.add_argument("--compact", action="store_true",
                        help="Export smaller model-only checkpoints; keep original full checkpoints untouched.")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    stage1 = (args.stage1.expanduser().resolve() if args.stage1 else
              newest((root / "output/stage1_local_5000").rglob("checkpoint_5000.pth"), "第一阶段最终权重"))
    stage2 = root / "output/stage2_local_5000/checkpoint_last.pth"
    for path in (stage1, stage2):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise SystemExit(f"检查点不存在或不在项目目录内：{path}")
    eval_files = {role: newest((root / "output" / name).rglob("evaluate.txt"), f"{role} 评估结果")
                  for role, name in {
                      "stage2": "vqav2_stage2_holdout_n1000_seed42",
                      "official": "vqav2_official_holdout_n1000_seed42"}.items()}
    warnings = []
    data = {"date": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d"),
            "project_root": str(root), "scores": {role: last_metrics(path) for role, path in eval_files.items()},
            "score_source": "归档脚本直接读取的服务器 evaluate.txt",
            "metric_files": {role: str(path) for role, path in eval_files.items()}, "warnings": warnings}
    for role, score in data["scores"].items():
        if any(abs(score[k] - EXPECTED[role][k]) > 0.005 for k in EXPECTED[role]):
            warnings.append(f"{role} 最新结果与对话截图不同，报告采用服务器实际文件，请核对运行目录。")

    subset = root / "data/coco/annotations/holdout_n1000_seed42"
    subset_input = subset / "vqa_val_eval.json"
    if not subset_input.is_file():
        raise SystemExit(f"缺少 1000 题评估输入：{subset_input}")
    selected_rows = json.loads(subset_input.read_text(encoding="utf-8"))
    qids = sorted(int(row["question_id"]) for row in selected_rows)
    if len(qids) != 1000 or len(set(qids)) != 1000:
        raise SystemExit("题集数量或唯一 question_id 数量不是 1000，请检查实际评估配置。")
    training_path = root / "data/coco/annotations/coco_stage1_existing_images.json"
    if not training_path.is_file():
        raise SystemExit(f"缺少需要保留的训练标注：{training_path}")
    training_rows = json.loads(training_path.read_text(encoding="utf-8"))
    training_names = {row["image"].replace("\\", "/").rsplit("/", 1)[-1] for row in training_rows}
    if any(row["image"].replace("\\", "/").rsplit("/", 1)[-1] in training_names for row in selected_rows):
        raise SystemExit("发现评估图片与本次训练图片交叉，请核对使用的题集。")
    data["question_count"] = len(qids)
    data["question_ids_sha256"] = hashlib.sha256(json.dumps(qids).encode()).hexdigest()
    data["training_data"] = {"images": len(training_names), "caption_pairs": len(training_rows)}
    del training_rows, selected_rows

    print("只在 CPU 上读取检查点元信息，不启动训练……", flush=True)
    s2 = checkpoint_metadata(stage2, warnings)
    recorded_stage1 = Path(s2["stage1"]).expanduser().resolve() if s2.get("stage1") else None
    if recorded_stage1 and not args.stage1:
        if recorded_stage1.is_file() and recorded_stage1.is_relative_to(root):
            stage1 = recorded_stage1
            print(f"按第二阶段元信息保存其实际第一阶段来源：{stage1}", flush=True)
        else:
            warnings.append("第二阶段记录的第一阶段来源路径不可访问，保存当前选中的第一阶段结果，请核对来源。")
    if recorded_stage1 and args.stage1 and recorded_stage1 != stage1:
        warnings.append("手动指定的第一阶段路径与第二阶段元信息记录不同，请核对模型来源。")
    data["checkpoint_metadata"] = {"stage2": s2, "stage1": checkpoint_metadata(stage1, warnings)}
    if s2 and (s2.get("kind") != "blip2_local_stage2_v1" or s2.get("step") != 5000):
        warnings.append("第二阶段检查点类型或步数不符合已完成 5000 步的预期，请核对。")
    data["training_log_summary"] = log_summary(root / "output/stage2_local_5000/train_log.jsonl")
    if not data["training_log_summary"]:
        warnings.append("未能读取第二阶段训练日志；报告不填入 loss 和收敛结论。")
    weight_sources = {"stage1": stage1, "stage2": stage2}
    if args.compact:
        weight_sources = export_compact(root, weight_sources)
        data["export_mode"] = "model_only"
        print("精简权重用于重载模型，不含原优化器状态，不能精确续训。", flush=True)
    else:
        data["export_mode"] = "full_checkpoints"

    paths = set()
    def add(path):
        if path.is_file() and path.resolve().is_relative_to(root):
            paths.add(path)

    for pattern in ("*.py", "*.yaml", "*.yml", "*.md", "constraints.txt"):
        for path in root.glob(pattern):
            add(path)
    for folder in (stage1.parent, stage2.parent, *(p.parent for p in eval_files.values()), subset):
        for path in folder.rglob("*"):
            if path.suffix.lower() in ALLOWED:
                add(path)
    add(training_path)
    add(root / "data/coco/annotations/answer_list.json")
    source = root / f"LAVIS-{COMMIT}"
    # Preserve the exact small source files, including the stage-1 factory fix.
    for path in source.rglob("*"):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
            if path.suffix.lower() in {".py", ".yaml", ".yml", ".toml", ".txt"} or path.name.startswith("LICENSE") or path.name.endswith(".before_stage1"):
                add(path)
    if not (source / "train.py").is_file():
        warnings.append("固定版本源码目录不存在，未能归档源码；未来恢复需要重新取得该提交。")
    # Small local model configs/tokenizers aid recovery; never include .bin/.pth weights.
    for model_dir in (root / "models/bert-base-uncased", root / "models/flan-t5-xl"):
        for path in model_dir.glob("*"):
            if path.suffix.lower() in {".json", ".txt", ".model"}:
                add(path)
    paths = sorted(paths)
    if any(path.stat().st_size > 256 * 1024**2 for path in paths):
        raise SystemExit("发现超过 256 MiB 的小文件候选；已停止，避免意外打包大文件。")
    required = sum(path.stat().st_size for path in paths) + 20 * 1024**2
    if shutil.disk_usage(root).free < required:
        raise SystemExit(f"小文件打包保守估计需要 {required / 1024**2:.1f} MiB 空闲空间。")

    entries = []
    for role, path in weight_sources.items():
        print(f"校验 {role} 权重：{path}（{path.stat().st_size / 1024**3:.2f} GiB）", flush=True)
        entries.append({"role": role, "server_path": str(path),
                        "local_path": f"checkpoints/{role}/{path.name}",
                        "size": path.stat().st_size, "sha256": sha256_file(path),
                        "included_in_zip": False})
    for path in paths:
        entries.append({"role": "evidence", "server_path": str(path),
                        "local_path": "project/" + path.relative_to(root).as_posix(),
                        "size": path.stat().st_size, "sha256": sha256_file(path),
                        "included_in_zip": True})

    base_files = []
    for model_dir in (root / "models/bert-base-uncased", root / "models/flan-t5-xl",
                      root / "torch-cache/hub/checkpoints"):
        for path in model_dir.glob("*"):
            if path.is_file() and path.suffix.lower() in {".bin", ".pth", ".safetensors"}:
                base_files.append({"server_path": str(path), "size": path.stat().st_size,
                                   "included_in_zip": False})
    data["base_model_inventory"] = base_files
    environment = (f"Executable: {sys.executable}\nPython: {sys.version}\n"
                   f"prefix: {sys.prefix}\nbase_prefix: {sys.base_prefix}\n"
                   f"LAVIS commit: {COMMIT}\n")
    print("记录 Python/pip 环境……", flush=True)
    pip_freeze = capture([sys.executable, "-m", "pip", "freeze", "--all"])
    print("记录 GPU 驱动与 conda 环境（只查询，不安装软件）……", flush=True)
    gpu_info = capture(["nvidia-smi"])
    conda = shutil.which("conda")
    conda_info = capture([conda, "list", "--explicit", "-p", sys.base_prefix]) if conda else "conda executable unavailable\n"
    first_name = weight_sources["stage1"].name
    second_name = weight_sources["stage2"].name
    readme = f"""保存目录结构：
本目录/
  BLIP-2_小规模复现实验报告.md
  MANIFEST.json
  verify_backup.py
  project/                  原配置、代码、日志、标注和预测结果
  checkpoints/stage1/       手动下载第一阶段 {first_name} 到这里
  checkpoints/stage2/       手动下载第二阶段 {second_name} 到这里

在 Windows PowerShell 中运行：python .\\verify_backup.py
出现 ALL FILES VERIFIED 表示清单中的小文件和两个检查点均通过 SHA256 校验。
基础模型和 COCO 图片不在包内；基础模型文件名及大小见 experiment_summary.json。
环境清单用于重建，不能将 Linux 的 .venv 直接复制到 Windows 使用。
report 中的日期、指标、检查点元信息和日志摘要取自打包时的服务器状态。
训练／评估配置才是实际运行参数的最终证据。打包不会更改或删除原始结果。
"""
    if args.compact:
        readme += "本包配套的是 model_only 精简参数，不能恢复原优化器状态并精确续训；完整原件保留在服务器。\n"
    generated = {
        "experiment_summary.json": json.dumps(data, ensure_ascii=False, indent=2),
        "question_ids.json": json.dumps(qids),
        "BLIP-2_小规模复现实验报告.md": make_report(data),
        "verify_backup.py": VERIFY_SCRIPT,
        "README_保存说明.txt": readme,
        "environment/python.txt": environment,
        "environment/pip_freeze.txt": pip_freeze,
        "environment/nvidia-smi.txt": gpu_info,
        "environment/conda_explicit.txt": conda_info,
    }
    for name, content in generated.items():
        encoded = content.encode("utf-8")
        entries.append({"role": "generated", "local_path": name, "size": len(encoded),
                        "sha256": hashlib.sha256(encoded).hexdigest(), "included_in_zip": True})
    manifest = {"format": "blip2_results_v1", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "source_commit": COMMIT, "files": entries,
                "note": "Weights are not inside the ZIP. Download the two listed checkpoint files separately."}
    output = root / "final_archive"
    output.mkdir(exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    archive = output / f"blip2_results_{stamp}.zip"
    temporary = archive.with_suffix(".zip.part")
    print(f"打包 {len(paths)} 个小文件，权重保持原位、分别下载……", flush=True)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for entry in entries:
            if entry["included_in_zip"] and entry["role"] != "generated":
                bundle.write(entry["server_path"], entry["local_path"])
        bundle.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, content in generated.items():
            bundle.writestr(name, content)
    with zipfile.ZipFile(temporary) as bundle:
        bad = bundle.testzip()
        if bad:
            raise SystemExit(f"ZIP 校验失败：{bad}；原始文件保持不变。")
    temporary.replace(archive)
    print(f"\n证据包完成：{archive}（{archive.stat().st_size / 1024**2:.1f} MiB）", flush=True)
    print("还必须分别下载以下两个权重，服务器路径 → 本地相对路径：", flush=True)
    for entry in entries[:2]:
        print(f"{entry['server_path']}\n  → {entry['local_path']}", flush=True)
    print("解压证据包后按上述结构放好权重，运行 python verify_backup.py 核对。", flush=True)
    for item in warnings:
        print(f"待核对：{item}", flush=True)


if __name__ == "__main__":
    main()
