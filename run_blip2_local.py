"""Run the pinned LAVIS BLIP-2 FlanT5-XL model using uploaded local files."""

import argparse
import json
import os
from pathlib import Path
import sys
import tarfile
import zipfile


def validate_stage2_state(state, expected_shapes):
    missing = sorted(set(expected_shapes) - set(state))
    extra = sorted(set(state) - set(expected_shapes))
    wrong = [name for name, shape in expected_shapes.items()
             if name in state and tuple(state[name].shape) != tuple(shape)]
    if missing or extra or wrong:
        raise SystemExit(
            f"第二阶段参数不完整或不兼容：缺少 {missing[:5]}；"
            f"额外 {extra[:5]}；形状不符 {wrong[:5]}"
        )


def check_checkpoint_format(path):
    """Reject unreadable checkpoint containers before allocating model memory."""
    with path.open("rb") as stream:
        header = stream.read(16)

    if header.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                if not any(name.endswith("data.pkl") for name in archive.namelist()):
                    raise SystemExit(f"ZIP 中没有 PyTorch 的 data.pkl：{path}")
        except zipfile.BadZipFile as error:
            raise SystemExit(f"权重 ZIP 结构损坏或上传未完成：{path}") from error
    elif not header.startswith(b"\x80") and not tarfile.is_tarfile(path):
        raise SystemExit(
            f"权重文件头无法识别：{path}\n"
            f"前 16 字节：{header.hex(' ')}\n"
            "请比较本机与服务器文件的 MD5，确认下载及上传是否完整。"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="Check uploaded files without loading the model."
    )
    parser.add_argument("--checkpoint", type=Path, help="Use your stage-2 checkpoint instead of official weights.")
    parser.add_argument("--image", type=Path, help="Optional local image path.")
    parser.add_argument("--prompt", default="Question: Which city is this? Short answer:")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source = root / "LAVIS-8491f1d333cce678b1230cb940fb249253f3ea4d"
    models = root / "models"
    bert = models / "bert-base-uncased"
    t5 = models / "flan-t5-xl"
    checkpoints = root / "torch-cache" / "hub" / "checkpoints"
    eva_weight = checkpoints / "eva_vit_g.pth"
    blip_weight = (args.checkpoint.expanduser().resolve() if args.checkpoint
                   else checkpoints / "blip2_pretrained_flant5xl.pth")
    model_config = source / "lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml"
    image_path = (args.image.expanduser().resolve() if args.image
                  else source / "docs/_static/merlion.png")

    required = [source / "setup.py", model_config, image_path, eva_weight, blip_weight]
    required.extend(
        bert / name
        for name in ["config.json", "tokenizer_config.json", "vocab.txt", "pytorch_model.bin"]
    )
    required.extend(
        t5 / name
        for name in [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "spiece.model",
            "pytorch_model.bin.index.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ]
    )

    missing = [path for path in required if not path.is_file()]
    if missing:
        print("缺少以下文件，请按这些路径补齐：", flush=True)
        for path in missing:
            print(path)
        raise SystemExit(1)

    for path in required:
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))

    weights = [
        eva_weight,
        blip_weight,
        bert / "pytorch_model.bin",
        t5 / "pytorch_model-00001-of-00002.bin",
        t5 / "pytorch_model-00002-of-00002.bin",
    ]
    for path in weights:
        size = path.stat().st_size
        if size < 1_000_000:
            raise SystemExit(f"权重文件过小，请检查是否误下载了网页或指针文件：{path}")
        check_checkpoint_format(path)
        print(f"权重文件 {path.name}: {size / 1024**3:.3f} GiB", flush=True)

    index = json.loads((t5 / "pytorch_model.bin.index.json").read_text(encoding="utf-8"))
    for name in sorted(set(index["weight_map"].values())):
        if not (t5 / name).is_file():
            raise SystemExit(f"索引引用的分片不存在：{t5 / name}")

    print("文件路径、配置及权重容器格式检查通过。", flush=True)
    if args.check_only:
        return

    # Set these before importing Transformers or Torch so caches and offline mode apply.
    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["TORCH_HOME"] = str(root / "torch-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # This LAVIS revision uses the relative name "bert-base-uncased" internally.
    # Running from models makes it resolve to the uploaded local BERT directory.
    os.chdir(models)
    sys.path.insert(0, str(source))

    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from lavis.models import load_preprocess
    from lavis.models.blip2_models.blip2_t5 import Blip2T5

    if not torch.cuda.is_available():
        raise SystemExit("当前 Python 无法使用 CUDA，请检查 .venv 激活情况。")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    config = OmegaConf.load(model_config)
    config.model.t5_model = str(t5)
    config.model.pretrained = str(blip_weight)
    config.model.load_finetuned = False

    print("正在读取本地 EVA、BERT、FlanT5-XL 和 BLIP-2 权重……", flush=True)
    if args.checkpoint:
        payload = torch.load(blip_weight, map_location="cpu", weights_only=False)
        if payload.get("kind") != "blip2_local_stage2_v1":
            raise SystemExit("--checkpoint 需要 train_blip2_stage2_local.py 保存的第二阶段权重。")
        step, total = payload["step"], payload["total_steps"]
        state = payload["model"]
        del payload  # Discard optimizer state before constructing the model.
        model = Blip2T5(img_size=int(config.model.image_size),
                       num_query_token=int(config.model.num_query_token),
                       t5_model=str(t5), freeze_vit=True, max_txt_len=32)
        validate_stage2_state(state, {name: p.shape for name, p in model.named_parameters()
                                      if p.requires_grad})
        model.load_state_dict(state, strict=False)
        del state
        print(f"已严格检查并加载你训练的第二阶段参数：{step}/{total} 步。", flush=True)
        print(f"使用的权重：{blip_weight}", flush=True)
    else:
        model = Blip2T5.from_config(config.model)
    print("权重加载完成，正在移动到 GPU……", flush=True)
    model = model.eval().to("cuda")
    processors, _ = load_preprocess(config.preprocess)

    image = Image.open(image_path).convert("RGB")
    pixels = processors["eval"](image).unsqueeze(0).to("cuda")
    prompt = args.prompt
    print("正在生成答案……", flush=True)
    with torch.inference_mode():
        answer = model.generate({"image": pixels, "prompt": prompt})

    print("问题:", prompt)
    print("模型回答:", answer)
    print("峰值显存 GiB:", round(torch.cuda.max_memory_allocated() / 1024**3, 2))


if __name__ == "__main__":
    main()
