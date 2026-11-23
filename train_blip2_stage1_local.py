"""Train BLIP-2 stage 1 on Karpathy-training captions for existing COCO images."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


COMMIT = "8491f1d333cce678b1230cb940fb249253f3ea4d"
ANNOTATION_NAME = "coco_stage1_existing_images.json"


def respect_pretraining_flag(source):
    """Make the pinned factory honor its existing load_pretrained configuration."""
    path = source / "lavis/models/blip2_models/blip2_qformer.py"
    text = path.read_text(encoding="utf-8")
    original = "        model.load_checkpoint_from_config(cfg)\n"
    fixed = (
        '        if cfg.get("load_pretrained", True):\n'
        "            model.load_checkpoint_from_config(cfg)\n"
    )
    if fixed in text:
        print("load_pretrained 开关已修正。", flush=True)
        return
    if text.count(original) != 1:
        raise SystemExit(f"Q-Former 源码与预期版本不同，请检查：{path}")
    backup = path.with_name(path.name + ".before_stage1")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(original, fixed, 1), encoding="utf-8")
    print(f"已修正 load_pretrained 开关，原文件备份于 {backup}", flush=True)


def make_config(root, annotation, steps, batch_size):
    """Use the official model, dataset builder, losses, optimizer, and iteration runner."""
    images = root / "data/coco/images"
    return {
        "model": {
            "arch": "blip2",
            "model_type": "pretrain",
            "load_pretrained": False,
            "load_finetuned": False,
            "freeze_vit": True,
        },
        "datasets": {
            "coco_caption": {
                "vis_processor": {
                    "train": {"name": "blip2_image_train", "image_size": 224},
                },
                "text_processor": {"train": {"name": "blip_caption"}},
                "build_info": {
                    "images": {"storage": str(images)},
                    "annotations": {
                        "train": {"url": str(annotation), "storage": str(annotation)},
                        # Replace the builder's default validation/test annotations with
                        # local paths so it does not download unused caption splits.
                        "val": {"url": str(annotation), "storage": str(annotation)},
                        "test": {"url": str(annotation), "storage": str(annotation)},
                    },
                },
            },
        },
        "run": {
            "task": "image_text_pretrain",
            "runner": "runner_iter",
            "lr_sched": "linear_warmup_cosine_lr",
            "init_lr": 1e-4,
            "min_lr": 1e-5,
            "warmup_lr": 1e-6,
            "warmup_steps": min(100, max(1, steps // 5)),
            "weight_decay": 0.05,
            "beta2": 0.98,
            "max_epoch": 1,
            "max_iters": steps,
            "iters_per_inner_epoch": steps,
            "batch_size_train": batch_size,
            "batch_size_eval": batch_size,
            "accum_grad_iters": 1,
            "num_workers": 4,
            "log_freq": 10,
            "seed": 42,
            "output_dir": str(root / f"output/stage1_local_{steps}"),
            "amp": True,
            "resume_ckpt_path": None,
            "evaluate": False,
            "train_splits": ["train"],
            "valid_splits": [],
            "test_splits": [],
            "device": "cuda",
            "world_size": 1,
            "dist_url": "env://",
            "distributed": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2:
        parser.error("--steps must be positive; --batch-size must be at least 2 for negatives")

    root = Path(__file__).resolve().parent
    source = root / f"LAVIS-{COMMIT}"
    images = root / "data/coco/images"
    annotations = root / "data/coco/annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    annotation = annotations / ANNOTATION_NAME
    if not annotation.is_file():
        bundle = root / "coco_stage1_existing_images.zip"
        if not bundle.is_file():
            raise SystemExit(f"请上传训练标注包：{bundle}")
        with zipfile.ZipFile(bundle) as archive:
            archive.extract(ANNOTATION_NAME, annotations)

    rows = json.loads(annotation.read_text(encoding="utf-8"))
    image_names = {row["image"] for row in rows}
    missing = [str(images / name) for name in sorted(image_names) if not (images / name).is_file()]
    if missing:
        raise SystemExit(f"缺少 {len(missing)} 张训练图片，例如：\n" + "\n".join(missing[:5]))
    print(f"训练数据：{len(image_names)} 张图片，{len(rows)} 条图文对。", flush=True)

    required = [
        source / "train.py",
        root / "models/bert-base-uncased/pytorch_model.bin",
        root / "models/bert-base-uncased/config.json",
        root / "models/bert-base-uncased/vocab.txt",
        root / "torch-cache/hub/checkpoints/eva_vit_g.pth",
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"缺少本地文件：{path}")

    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["TORCH_HOME"] = str(root / "torch-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    from omegaconf import OmegaConf

    respect_pretraining_flag(source)
    config = make_config(root, annotation, args.steps, args.batch_size)
    config_path = root / f"stage1_local_{args.steps}.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    print(f"训练配置：{config_path}", flush=True)
    print("初始化使用本地 BERT 和 EVA；视觉编码器冻结，优化 Q-Former、query 与任务头。", flush=True)
    if args.prepare_only:
        return

    print(f"开始第一阶段训练：{args.steps} 个训练迭代，batch size {args.batch_size}。", flush=True)
    command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=1",
        str(source / "train.py"),
        "--cfg-path", str(config_path),
    ]
    result = subprocess.run(command, cwd=root / "models")
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f"训练结束，检查 {root / f'output/stage1_local_{args.steps}'} 中的日志和 checkpoint。", flush=True)


if __name__ == "__main__":
    main()
