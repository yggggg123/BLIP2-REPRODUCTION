"""Single-A800, offline BLIP-2/Flan-T5-XL stage-2 training.

Uses the pinned LAVIS model's prefix-LM forward. Captions are split at a
random word boundary (a local implementation choice). This is a small-data
experiment, not the paper's 129M-image pretraining run. Checkpoints contain
trainable weights and optimizer state; frozen EVA/T5 weights stay separate.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time


COMMIT = "8491f1d333cce678b1230cb940fb249253f3ea4d"
KIND = "blip2_local_stage2_v1"


def split_caption(caption, rng=random):
    words = caption.split()
    if not words:
        raise ValueError("Empty caption cannot be a generation target")
    cut = rng.randint(1, len(words) - 1) if len(words) > 1 else 0
    return " ".join(words[:cut]), " ".join(words[cut:])


def learning_rate(step, total, warmup):
    # step is zero-based; cosine decay is applied at every optimizer update.
    if step < warmup:
        return 1e-6 + (1e-4 - 1e-6) * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return 5e-5 + 0.5 * (1e-4 - 5e-5) * (1 + math.cos(math.pi * progress))


def choose_stage1(root, explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        candidates = list((root / "output/stage1_local_5000").rglob("checkpoint_5000.pth"))
        if not candidates:
            raise SystemExit("找不到第一阶段 checkpoint_5000.pth，请用 --stage1 指定其绝对路径。")
        path = max(candidates, key=lambda item: item.stat().st_mtime)
        if len(candidates) > 1:
            print("发现多个第一阶段结果，使用修改时间最新的一个；可用 --stage1 指定。", flush=True)
    if not path.is_file():
        raise SystemExit(f"第一阶段权重不存在：{path}")
    return path


def select_stage1_weights(state, expected_shapes):
    """Require every surviving stage-1 trainable tensor, except the new FC layer."""
    required = {name: shape for name, shape in expected_shapes.items()
                if not name.startswith("t5_proj.")}
    missing = sorted(set(required) - set(state))
    wrong = [name for name, shape in required.items()
             if name in state and tuple(state[name].shape) != tuple(shape)]
    if missing or wrong:
        raise ValueError(f"第一阶段权重不兼容：缺少 {missing[:8]}；形状不符 {wrong[:8]}")
    return {name: state[name] for name in required}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", help="Path to your stage-1 checkpoint; default: newest 5000-step run")
    parser.add_argument("--steps", type=int, default=5000, help="Planned total optimizer updates")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stop-after", type=int, help="Stop and save at this absolute step, e.g. 100")
    parser.add_argument("--resume", help="Resume this script's stage-2 checkpoint, including optimizer")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every) < 1 or args.workers < 0:
        parser.error("steps/batch-size/save-every must be positive; workers must be nonnegative")
    stop = args.steps if args.stop_after is None else args.stop_after
    if not 1 <= stop <= args.steps:
        parser.error("stop-after must be between 1 and steps")

    root = Path(__file__).resolve().parent
    source = root / f"LAVIS-{COMMIT}"
    annotation = root / "data/coco/annotations/coco_stage1_existing_images.json"
    t5 = root / "models/flan-t5-xl"
    resume = Path(args.resume).expanduser().resolve() if args.resume else None
    stage1 = None if resume else choose_stage1(root, args.stage1)
    required = [source / "lavis/models/blip2_models/blip2_t5.py", annotation,
                root / "torch-cache/hub/checkpoints/eva_vit_g.pth"]
    required += [root / "models/bert-base-uncased" / name
                 for name in ("config.json", "vocab.txt", "pytorch_model.bin")]
    required += [t5 / name for name in ("config.json", "tokenizer_config.json",
                                       "tokenizer.json", "spiece.model",
                                       "pytorch_model.bin.index.json")]
    if resume:
        required.append(resume)
    for path in required:
        if not path.is_file():
            raise SystemExit(f"缺少本地文件：{path}")
    index = json.loads((t5 / "pytorch_model.bin.index.json").read_text(encoding="utf-8"))
    for name in set(index["weight_map"].values()):
        if not (t5 / name).is_file():
            raise SystemExit(f"缺少 T5 分片：{t5 / name}")
    rows = json.loads(annotation.read_text(encoding="utf-8"))
    if not rows or any(not row["caption"].strip() for row in rows):
        raise SystemExit("标注为空或含空 caption，请检查训练标注。")
    image_names = {row["image"] for row in rows}
    missing = [name for name in image_names if not (root / "data/coco/images" / name).is_file()]
    if missing:
        raise SystemExit(f"缺少 {len(missing)} 张图片，例如：{missing[:3]}")
    print(f"沿用训练集：{len(image_names)} 张图片、{len(rows)} 条 caption。", flush=True)
    print(f"权重来源：{resume or stage1}", flush=True)
    print(f"计划 {args.steps} 步，batch={args.batch_size}；本次停在第 {stop} 步。", flush=True)
    print(f"磁盘剩余 {shutil.disk_usage(root).free / 1024**3:.2f} GiB。", flush=True)
    del rows
    if args.prepare_only:
        print("路径检查通过；尚未加载权重或验证 GPU。", flush=True)
        return

    os.environ.update(HF_HOME=str(root / "huggingface"), TORCH_HOME=str(root / "torch-cache"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    os.chdir(root / "models")  # Pinned LAVIS resolves bert-base-uncased relatively.
    sys.path.insert(0, str(source))

    import torch
    from torch.utils.data import DataLoader
    from lavis.datasets.datasets.caption_datasets import CaptionDataset
    from lavis.models.blip2_models.blip2_t5 import Blip2T5
    from lavis.processors.blip_processors import BlipCaptionProcessor, Blip2ImageTrainProcessor

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("需要支持 BF16 的 CUDA GPU，请激活原来的 .venv 并检查 A800。")
    torch.set_num_threads(8)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    class PrefixDataset(CaptionDataset):
        def __getitem__(self, item):
            sample = super().__getitem__(item)
            sample["text_input"], sample["text_output"] = split_caption(sample["text_input"])
            return sample

    dataset = PrefixDataset(Blip2ImageTrainProcessor(image_size=224),
                            BlipCaptionProcessor(max_words=32),
                            str(root / "data/coco/images"), [str(annotation)])
    if len(dataset) < args.batch_size:
        raise SystemExit("训练样本数小于 batch-size。")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True)
    print("加载本地 EVA、BERT、Flan-T5-XL……", flush=True)
    # Construct directly: never load the official completed stage-2 adapter here.
    model = Blip2T5(img_size=224, num_query_token=32, t5_model=str(t5),
                   freeze_vit=True, max_txt_len=32)
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    allowed = ("Qformer.", "query_tokens", "ln_vision.", "t5_proj.")
    if any(not name.startswith(allowed) for name in trainable):
        raise RuntimeError("检测到预期以外的可训练模块，停止训练。")

    checkpoint = torch.load(str(resume or stage1), map_location="cpu", weights_only=False)
    if resume:
        if checkpoint.get("kind") != KIND:
            raise SystemExit("--resume 只接受本脚本保存的第二阶段权重。")
        if checkpoint.get("inference_only") or "optimizer" not in checkpoint:
            raise SystemExit("这是精简模型权重，不含优化器状态；--resume 需要原始完整检查点。")
        if checkpoint["total_steps"] != args.steps or checkpoint["batch_size"] != args.batch_size:
            raise SystemExit("续训时 steps 和 batch-size 必须与原计划一致。")
        if set(checkpoint["model"]) != set(trainable):
            raise SystemExit("续训权重中的可训练参数集合与当前模型不一致。")
        model.load_state_dict(checkpoint["model"], strict=False)
        completed = checkpoint["step"]
    else:
        weights = select_stage1_weights(checkpoint["model"],
                                       {name: p.shape for name, p in trainable.items()})
        model.load_state_dict(weights, strict=False)
        completed = 0
        print(f"已接入第一阶段 {len(weights)} 个参数张量；新建 t5_proj 投影层。", flush=True)
        del weights
    if completed >= stop:
        print(f"权重已训练至 {completed} 步，本次目标无需继续。", flush=True)
        return
    model = model.to("cuda").train()
    parameters = dict(model.named_parameters())
    groups = [
        {"params": [parameters[n] for n in trainable if parameters[n].ndim >= 2
                    and "bias" not in n and "norm" not in n], "weight_decay": 0.05},
        {"params": [parameters[n] for n in trainable if parameters[n].ndim < 2
                    or "bias" in n or "norm" in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.98))
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    provenance = checkpoint.get("stage1", str(stage1))
    del checkpoint
    count = sum(p.numel() for p in trainable.values())
    print(f"GPU：{torch.cuda.get_device_name(0)}；可训练参数：{count / 1e6:.2f} M。", flush=True)
    print("EVA 与 Flan-T5-XL 冻结；更新 Q-Former、query、视觉 LayerNorm 和 t5_proj。", flush=True)
    print("使用 BF16 autocast；随机按词切分 caption，预测后缀。", flush=True)

    output = root / f"output/stage2_local_{args.steps}"
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "checkpoint_last.pth"
    if last_path.exists() and not resume:
        raise SystemExit(f"已有第二阶段结果，请用 --resume {last_path} 继续，避免覆盖。")
    # Two copies during atomic replacement, each with FP32 weights + Adam moments.
    required_space = count * 12 * 2 + 512 * 1024**2
    if shutil.disk_usage(root).free < required_space:
        raise SystemExit(f"保存检查点预计需至少 {required_space / 1024**3:.1f} GiB 空闲磁盘。")

    def save(step):
        state = {name: p.detach().cpu() for name, p in model.named_parameters() if name in trainable}
        payload = dict(kind=KIND, model=state, optimizer=optimizer.state_dict(),
                       step=step, total_steps=args.steps, batch_size=args.batch_size,
                       stage1=provenance, python_rng=random.getstate(),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                       data_scope="COCO Karpathy train captions for existing val2014 images",
                       prefix_split="uniform random internal word boundary; max_words=32")
        temporary = output / "checkpoint_last.pth.tmp"
        torch.save(payload, temporary)
        temporary.replace(last_path)
        print(f"已保存第 {step} 步：{last_path}", flush=True)

    # Resume preserves updates/LR/optimizer, but starts a new shuffled data pass;
    # it does not promise bit-identical replay of prefetched worker batches.
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    warmup = min(100, max(1, args.steps // 5))
    start = time.monotonic()
    initial_step = completed
    torch.cuda.reset_peak_memory_stats()
    with (output / "train_log.jsonl").open("a", encoding="utf-8") as log:
        for step in range(completed, stop):
            try:
                samples = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                samples = next(iterator)
            samples["image"] = samples["image"].to("cuda", non_blocking=True)
            lr = learning_rate(step, args.steps, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(samples)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"第 {step + 1} 步 loss 非有限值；最近保存权重为 {last_path}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(groups[0]["params"] + groups[1]["params"],
                                                   max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step + 1
            if completed == initial_step + 1 or completed % 10 == 0 or completed == stop:
                loss_value = loss.item()
                elapsed = time.monotonic() - start
                seconds_per_step = elapsed / (completed - initial_step)
                row = dict(step=completed, loss=loss_value, lr=lr, grad_norm=norm.item(),
                           seconds_per_step=seconds_per_step,
                           peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"Stage2 [{completed}/{args.steps}] loss={loss_value:.4f} lr={lr:.7f} "
                      f"time={seconds_per_step:.2f}s/step peak={row['peak_gpu_gib']:.2f}GiB "
                      f"eta={(stop-completed)*seconds_per_step/60:.1f}min", flush=True)
            if completed % args.save_every == 0 or completed == stop:
                save(completed)
    print(f"本次完成至 {completed}/{args.steps} 步。日志：{output / 'train_log.jsonl'}", flush=True)
    if completed < args.steps:
        print(f"继续：python -u {Path(__file__).resolve()} --steps {args.steps} "
              f"--batch-size {args.batch_size} --resume {last_path}", flush=True)


if __name__ == "__main__":
    main()
