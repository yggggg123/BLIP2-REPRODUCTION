"""Matched-budget VQA fine-tuning: plain, constant-condition gate, question gate.

Use --steps 1000 --stop-after 100 for a resumable smoke run. All variants
use the same immutable sample/answer plan and deterministic image resize. At the planned final
step the checkpoint is model-only to conserve disk; incomplete runs include
optimizer/RNG state for continuation. Existing experiment files are untouched.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def make_sample_plan(rows, steps, batch_size, seed):
    """Each item fixes row and vote-weighted answer, independent of model RNG."""
    if not rows or steps < 1 or batch_size < 1:
        raise ValueError("Empty training data or invalid training budget")
    if any(not row.get("answers") for row in rows):
        raise ValueError("Every training row needs its original answer votes")
    rng = random.Random(seed)
    plan = []
    while len(plan) < steps * batch_size:
        order = list(range(len(rows)))
        rng.shuffle(order)
        for index in order:
            plan.append([index, rng.randrange(len(rows[index]["answers"]))])
            if len(plan) == steps * batch_size:
                break
    return plan


def schedule_factor(step, total, warmup):
    if step < warmup:
        return 0.1 + 0.9 * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))


def choose_initial_checkpoint(root, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"初始化权重不存在：{path}")
        return path
    original = root / "output/stage2_local_5000/checkpoint_last.pth"
    if original.is_file():
        return original
    candidates = list((root / "compact_weights").rglob("stage2_model_only.pth"))
    candidates += list((root / "final_backup/checkpoints/stage2").glob("stage2_model_only.pth"))
    candidates = [path for path in candidates if path.is_file()]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise SystemExit("找不到你已训练的第二阶段权重；请使用 --init-checkpoint 指定。")


def checked_training_rows(root):
    folder = root / "data/coco/improvement_v1"
    manifest_path, train_path = folder / "split_manifest.json", folder / "train.json"
    if not manifest_path.is_file() or not train_path.is_file():
        raise SystemExit("请先运行 improvement/prepare_vqa_improvement.py 生成固定划分。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["outputs"]["train"]["sha256"]
    actual = file_sha256(train_path)
    if actual != expected:
        raise SystemExit("train.json 与划分清单哈希不一致，请检查，未启动训练。")
    rows = json.loads(train_path.read_text(encoding="utf-8"))
    if not rows or any(len(row.get("answers", [])) != 10 for row in rows):
        raise SystemExit("训练标注必须保留每题原始 10 个答案。")
    missing = [row["image"] for row in rows
               if not (root / "data/coco/images" / row["image"]).is_file()]
    if missing:
        raise SystemExit(f"训练图片缺失，例如：{missing[:3]}")
    return rows, actual, file_sha256(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--variant", required=True, choices=("plain", "constant", "question"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    stop = args.steps if args.stop_after is None else args.stop_after
    if min(args.steps, args.batch_size, args.save_every, args.hidden_dim) < 1 or args.workers < 0:
        parser.error("steps/batch-size/save-every/hidden-dim must be positive; workers >= 0")
    if not 1 <= stop <= args.steps or not 0 < args.rho < 1 or min(args.base_lr, args.gate_lr) <= 0:
        parser.error("invalid stop-after, rho, or learning rate")
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    root = args.root.expanduser().resolve()
    rows, train_hash, split_hash = checked_training_rows(root)
    plan = make_sample_plan(rows, args.steps, args.batch_size, args.seed)
    plan_text = json.dumps(plan, separators=(",", ":"))
    plan_hash = hashlib.sha256(plan_text.encode()).hexdigest()
    output = root / "output/improvement" / f"{args.variant}_seed{args.seed}_steps{args.steps}"
    last_path = output / "checkpoint_last.pth"
    resume = args.resume.expanduser().resolve() if args.resume else None
    if resume and not resume.is_file():
        raise SystemExit(f"续训文件不存在：{resume}")
    if not resume and output.exists() and any(output.iterdir()):
        raise SystemExit(f"已有该实验结果，请用 --resume 继续，或选择新的 seed/steps：{output}")
    init_path = None if resume else choose_initial_checkpoint(root, args.init_checkpoint)
    config = {
        "variant": args.variant, "seed": args.seed, "total_steps": args.steps,
        "batch_size": args.batch_size, "base_lr": args.base_lr, "gate_lr": args.gate_lr,
        "hidden_dim": args.hidden_dim, "rho": args.rho, "train_sha256": train_hash,
        "split_manifest_sha256": split_hash, "sample_plan_sha256": plan_hash,
        "warmup_steps": min(50, max(1, args.steps // 5)), "optimizer": "AdamW",
        "betas": [0.9, 0.98], "weight_decay": 0.05, "gradient_clip": 1.0,
        "image_size": 224, "max_question_tokens": 64, "max_answer_tokens": 16,
        "prompt": "Question: {} Short answer:", "target_sampling": "uniform among original 10 answer votes",
        "augmentation": "deterministic resize to 224 + normalize; no crop or flip for VQA labels",
        "frozen": ["visual_encoder", "t5_model"],
    }
    print(f"实验 {args.variant}；训练题数 {len(rows)}；本次目标 {stop}/{args.steps} 步。", flush=True)
    print(f"样本/答案计划 SHA256：{plan_hash}", flush=True)
    print(f"初始化/续训文件：{resume or init_path}", flush=True)
    print(f"输出：{output}；磁盘可用 {shutil.disk_usage(root).free / 1024**3:.2f} GiB", flush=True)
    if args.prepare_only:
        print("数据与运行参数检查通过；未加载模型，未启动 GPU 训练。", flush=True)
        return

    from question_gate import (configure_runtime, setup_model, load_improved_model,
                               checkpoint_payload)
    configure_runtime(root)
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from lavis.processors.blip_processors import BlipImageEvalProcessor, BlipQuestionProcessor

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("请激活原来的 .venv，并确认 A800/CUDA/BF16 可用。")
    torch.set_num_threads(8)
    # B and C must also start from identical random gate-down weights.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    completed = 0
    saved = None
    if resume:
        # Validate run identity before loading model or touching output files.
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("run_config") != config:
            raise SystemExit("续训参数、划分或样本计划与原实验不同，停止以免混用对照。")
        completed = int(saved["step"])
        if completed >= stop:
            print(f"已经完成至 {completed} 步，本次无需训练。", flush=True)
            return
        if saved.get("inference_only") or "optimizer" not in saved:
            raise SystemExit("该文件是最终精简权重，不包含续训优化器。")
        init_sha = saved["init_checkpoint_sha256"]
        init_source = saved["init_checkpoint"]
        del saved
        model, _ = load_improved_model(root, resume, device="cuda")
        saved = torch.load(resume, map_location="cpu", weights_only=False)
    else:
        print("计算初始权重 SHA256，并加载本地模型……", flush=True)
        init_sha = file_sha256(init_path)
        init_source = str(init_path)
        model = setup_model(root, variant=args.variant, init_checkpoint=init_path,
                            device="cuda", hidden_dim=args.hidden_dim, rho=args.rho)
    model.train()
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    allowed = ("Qformer.", "query_tokens", "ln_vision.", "t5_proj.", "channel_gate.")
    unexpected = [name for name in parameters if not name.startswith(allowed)]
    if unexpected:
        raise SystemExit(f"出现意外可训练模块：{unexpected[:5]}")
    groups = []
    for is_gate in (False, True):
        for decay in (False, True):
            selected = [p for name, p in parameters.items()
                        if name.startswith("channel_gate.") == is_gate
                        and (p.ndim >= 2 and "bias" not in name and "norm" not in name) == decay]
            if selected:
                lr = args.gate_lr if is_gate else args.base_lr
                groups.append({"params": selected, "lr": lr, "initial_lr": lr,
                               "weight_decay": 0.05 if decay else 0.0})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98))
    # Model/gate initialization must not change the training dropout stream.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    total_count = sum(p.numel() for p in parameters.values())
    gate_count = sum(p.numel() for n, p in parameters.items() if n.startswith("channel_gate."))
    print(f"可训练参数 {total_count / 1e6:.3f} M；其中门控 {gate_count} 个。", flush=True)
    # Existing checkpoint + replacement at incomplete save: model + Adam moments.
    required_free = total_count * (12 if completed else 16) + 256 * 1024**2
    if shutil.disk_usage(root).free < required_free:
        raise SystemExit(f"安全保存预计需要至少 {required_free / 1024**3:.2f} GiB 空闲空间。")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "sample_plan.json").write_text(plan_text, encoding="utf-8")
    # Cropping can remove queried objects; flipping changes left/right answers.
    processor = BlipImageEvalProcessor(image_size=224)
    question_processor = BlipQuestionProcessor(max_words=50)

    class PlannedDataset(Dataset):
        def __len__(self):
            return len(plan) - completed_at_start * args.batch_size

        def __getitem__(self, offset):
            row_idx, answer_idx = plan[completed_at_start * args.batch_size + offset]
            row = rows[row_idx]
            image = Image.open(root / "data/coco/images" / row["image"]).convert("RGB")
            pixels = processor(image)
            return {"image": pixels, "question": question_processor(row["question"]),
                    "answer": row["answers"][answer_idx], "question_id": row["question_id"]}

    completed_at_start = completed
    loader = DataLoader(PlannedDataset(), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))
    def save_checkpoint(step):
        # Only incomplete checkpoints retain optimizer/RNG; final files are compact.
        incomplete = step < args.steps
        if shutil.disk_usage(root).free < total_count * (12 if incomplete else 4) + 64 * 1024**2:
            raise RuntimeError("保存空间不足；原有检查点仍保留，请检查数据盘。")
        payload = checkpoint_payload(model, step=step, total_steps=args.steps, run_config=config,
                                     init_checkpoint=init_source, init_checkpoint_sha256=init_sha,
                                     inference_only=not incomplete)
        if incomplete:
            payload.update(optimizer=optimizer.state_dict(), python_rng=random.getstate(),
                           torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        temporary = output / "checkpoint_last.pth.part"
        torch.save(payload, temporary)
        temporary.replace(last_path)
        print(f"保存第 {step} 步：{last_path}（{'可续训' if incomplete else '最终精简模型'}）", flush=True)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with (output / "train_log.jsonl").open("a", encoding="utf-8") as log:
        for batch in loader:
            if completed >= stop:
                break
            step = completed
            batch["image"] = batch["image"].to("cuda", non_blocking=True)
            factor = schedule_factor(step, args.steps, config["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.forward_qa(batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"第 {step + 1} 步 loss 非有限值；请保留日志，不继续覆盖检查点。")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0,
                                                       error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed += 1
            if completed == completed_at_start + 1 or completed % 10 == 0 or completed == stop:
                row = {"step": completed, "loss": loss.item(), "grad_norm": grad_norm.item(),
                       "base_lr": args.base_lr * factor, "gate_lr": args.gate_lr * factor if gate_count else 0,
                       "seconds_per_step": (time.monotonic() - start) / (completed - completed_at_start),
                       "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"{args.variant} [{completed}/{args.steps}] loss={row['loss']:.4f} "
                      f"time={row['seconds_per_step']:.3f}s/step peak={row['peak_gpu_gib']:.2f}GiB", flush=True)
            if completed % args.save_every == 0 or completed == stop:
                save_checkpoint(completed)
    print(f"本次完成 {completed}/{args.steps} 步。", flush=True)
    if completed < args.steps:
        print(f"继续：python -u {Path(__file__).resolve()} --variant {args.variant} --steps {args.steps} "
              f"--batch-size {args.batch_size} --seed {args.seed} --resume {last_path}", flush=True)


if __name__ == "__main__":
    main()
