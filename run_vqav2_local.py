"""Prepare and run official LAVIS VQAv2 validation with local BLIP-2 weights."""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile


ANNOTATION_BASE = (
    "https://storage.googleapis.com/"
    "sfr-vision-language-research/LAVIS/datasets/vqav2/"
)
ANNOTATION_NAMES = [
    "vqa_val_eval.json",
    "answer_list.json",
    "v2_OpenEnded_mscoco_val2014_questions.json",
    "v2_mscoco_val2014_annotations.json",
]


def select_questions(records, training_rows=None, limit=0, seed=42):
    """Exclude complete training images, then sample questions reproducibly."""
    def image_name(row):
        return row["image"].replace("\\", "/").rsplit("/", 1)[-1]

    excluded = {image_name(row) for row in training_rows} if training_rows is not None else set()
    selected = [row for row in records if image_name(row) not in excluded]
    selected.sort(key=lambda row: int(row["question_id"]))
    if limit and limit < len(selected):
        selected = random.Random(seed).sample(selected, limit)
        selected.sort(key=lambda row: int(row["question_id"]))
    ids = [int(row["question_id"]) for row in selected]
    if not ids or len(ids) != len(set(ids)):
        raise SystemExit("评估问题为空或含重复 question_id，请检查标注。")
    return selected


def filter_coco_document(document, key, question_ids):
    selected = dict(document)
    selected[key] = [row for row in document[key] if int(row["question_id"]) in question_ids]
    ids = [int(row["question_id"]) for row in selected[key]]
    if len(ids) != len(question_ids) or set(ids) != question_ids:
        raise SystemExit(f"VQA {key} 与选取的 question_id 不匹配。")
    return selected


def download_annotation(opener, url, target):
    """Download JSON atomically with visible progress and bounded retries."""
    temporary = target.with_name(target.name + ".part")
    for attempt in range(1, 4):
        print(f"连接标注下载（第 {attempt}/3 次，单次网络读取超时 20 秒）：\n{url}", flush=True)
        received = 0
        started = time.monotonic()
        last_report = started
        try:
            with opener.open(url, timeout=20) as response, temporary.open("wb") as output:
                total = int(response.headers.get("Content-Length", 0))
                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        break
                    output.write(block)
                    received += len(block)
                    now = time.monotonic()
                    if received == len(block) or now - last_report >= 2:
                        size = f"{total / 1024**2:.1f} MiB" if total else "未知总大小"
                        speed = received / max(now - started, 0.001) / 1024**2
                        print(
                            f"{target.name}: {received / 1024**2:.1f} MiB / {size}，"
                            f"平均 {speed:.2f} MiB/s",
                            flush=True,
                        )
                        last_report = now
            print(f"下载完成，检查 JSON：{target.name}", flush=True)
            json.loads(temporary.read_text(encoding="utf-8"))
            temporary.replace(target)
            return
        except (OSError, ValueError) as error:
            print(f"本次下载失败：{error}", flush=True)
            if attempt == 3:
                raise SystemExit(
                    f"标注下载未完成，可在本机下载 {url}\n"
                    f"上传至 {target}，然后重新运行。"
                ) from error
            time.sleep(2)


def extract_with_space_check(archive_path, images):
    """Estimate additional disk usage from ZIP metadata before extracting."""
    with zipfile.ZipFile(archive_path) as archive:
        image_root = images.resolve()
        total_bytes = 0
        additional_bytes = 0
        for member in archive.infolist():
            target = images / member.filename
            if not target.resolve().is_relative_to(image_root):
                raise SystemExit(f"压缩包中存在异常路径：{member.filename}")
            total_bytes += member.file_size
            existing_bytes = target.stat().st_size if target.is_file() else 0
            additional_bytes += max(member.file_size - existing_bytes, 0)

        free_bytes = shutil.disk_usage(images).free
        print(f"ZIP 文件大小：{archive_path.stat().st_size / 1024**3:.2f} GiB", flush=True)
        print(f"解压后文件总大小：{total_bytes / 1024**3:.2f} GiB", flush=True)
        print(f"预计新增文件内容：{additional_bytes / 1024**3:.2f} GiB", flush=True)
        print(f"当前磁盘可用空间：{free_bytes / 1024**3:.2f} GiB", flush=True)
        if additional_bytes > free_bytes:
            raise SystemExit("磁盘空间不足，已在解压前停止。")

        print("开始逐文件解压 val2014.zip……", flush=True)
        archive.extractall(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data and config, then stop.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, help="Use your stage-2 checkpoint.")
    parser.add_argument("--exclude-training-images", action="store_true",
                        help="Exclude images in the local stage-1/stage-2 caption training set.")
    parser.add_argument("--limit", type=int, default=0, help="Question count; 0 means all eligible questions.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size must be positive; --limit must be nonnegative")

    root = Path(__file__).resolve().parent
    source = root / "LAVIS-8491f1d333cce678b1230cb940fb249253f3ea4d"
    original_config = source / "lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
    if not original_config.is_file():
        raise SystemExit(f"找不到固定版本的官方评估配置：{original_config}")
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint and not checkpoint.is_file():
        raise SystemExit(f"第二阶段权重不存在：{checkpoint}")
    # The local experiment used some val2014 images for caption training.
    # Automatically exclude these images when assessing our trained model.
    exclude_training = args.exclude_training_images or checkpoint is not None

    data = root / "data/coco"
    annotations = data / "annotations"
    images = data / "images"
    annotations.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    bundle = data / "vqav2_annotations.zip"
    if bundle.is_file():
        print("读取已上传的标注压缩包……", flush=True)
        with zipfile.ZipFile(bundle) as archive:
            for name in ANNOTATION_NAMES:
                if name not in archive.namelist():
                    raise SystemExit(f"标注压缩包缺少文件：{name}")
                if not (annotations / name).is_file():
                    archive.extract(name, annotations)

    # AutoDL's academic proxy covers Hugging Face; fetch Google-hosted annotations directly.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for name in ANNOTATION_NAMES:
        target = annotations / name
        if not target.is_file():
            url = ANNOTATION_BASE + name
            download_annotation(opener, url, target)
        else:
            json.loads(target.read_text(encoding="utf-8"))
            print(f"使用本地标注：{name}", flush=True)

    records = json.loads((annotations / ANNOTATION_NAMES[0]).read_text(encoding="utf-8"))
    full_count = len(records)
    training_rows = None
    if exclude_training:
        training_path = annotations / "coco_stage1_existing_images.json"
        if not training_path.is_file():
            raise SystemExit(f"需要训练标注以排除训练图片：{training_path}")
        training_rows = json.loads(training_path.read_text(encoding="utf-8"))
    records = select_questions(records, training_rows, args.limit, args.seed)
    del training_rows
    is_subset = exclude_training or args.limit > 0
    selection_name = (f"{'holdout' if exclude_training else 'sample'}_"
                      f"n{args.limit or 'all'}_seed{args.seed}")
    annotation_paths = [str(annotations / name) for name in ANNOTATION_NAMES]
    if is_subset:
        subset_dir = annotations / selection_name
        subset_dir.mkdir(exist_ok=True)
        question_ids = {int(row["question_id"]) for row in records}
        (subset_dir / ANNOTATION_NAMES[0]).write_text(json.dumps(records), encoding="utf-8")
        for name, key in [(ANNOTATION_NAMES[2], "questions"), (ANNOTATION_NAMES[3], "annotations")]:
            document = json.loads((annotations / name).read_text(encoding="utf-8"))
            filtered = filter_coco_document(document, key, question_ids)
            (subset_dir / name).write_text(json.dumps(filtered), encoding="utf-8")
            del document, filtered
        annotation_paths = [str(subset_dir / ANNOTATION_NAMES[0]), annotation_paths[1],
                            str(subset_dir / ANNOTATION_NAMES[2]), str(subset_dir / ANNOTATION_NAMES[3])]
        print(f"从 {full_count} 道题中选取 {len(records)} 道；排除本次训练图片：{exclude_training}。", flush=True)
        print("这是自定义评估子集，不能把得分直接与论文全量 VQAv2 分数比较。", flush=True)
    image_names = sorted({record["image"] for record in records})
    missing_images = [name for name in image_names if not (images / name).is_file()]

    if missing_images:
        archive_path = data / "val2014.zip"
        if archive_path.is_file():
            extract_with_space_check(archive_path, images)
            missing_images = [name for name in image_names if not (images / name).is_file()]

    if missing_images:
        examples = "\n".join(str(images / name) for name in missing_images[:5])
        raise SystemExit(
            f"还缺 {len(missing_images)} 张验证图片，例如：\n{examples}\n"
            "下载 http://images.cocodataset.org/zips/val2014.zip\n"
            f"上传至 {data / 'val2014.zip'}，然后重新运行。"
        )

    print(f"验证问题 {len(records)} 条，所需图片 {len(image_names)} 张，路径检查通过。", flush=True)

    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["TORCH_HOME"] = str(root / "torch-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    from omegaconf import OmegaConf

    if checkpoint and not args.prepare_only:
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("kind") != "blip2_local_stage2_v1":
            raise SystemExit("--checkpoint 必须是第二阶段训练脚本保存的权重。")
        if not all(name in payload["model"] for name in
                   ("query_tokens", "ln_vision.weight", "ln_vision.bias", "t5_proj.weight", "t5_proj.bias")):
            raise SystemExit("第二阶段权重缺少 query、视觉 LayerNorm 或 T5 投影参数。")
        print(f"评估你训练的参数：第 {payload['step']}/{payload['total_steps']} 步。", flush=True)
        del payload

    config = OmegaConf.load(original_config)
    config.model.t5_model = str(root / "models/flan-t5-xl")
    config.model.pretrained = str(checkpoint or
        root / "torch-cache/hub/checkpoints/blip2_pretrained_flant5xl.pth")
    config.model.load_finetuned = False
    config.datasets.coco_vqa.build_info = {
        "images": {"storage": str(images)},
        "annotations": {"val": {"url": annotation_paths, "storage": annotation_paths}},
    }
    config.run.batch_size_eval = args.batch_size
    run_name = (f"vqav2_{'stage2' if checkpoint else 'official'}_{selection_name}"
                if is_subset else "vqav2_local")
    config.run.output_dir = str(root / "output" / run_name)
    config_path = root / f"{run_name}.yaml"
    OmegaConf.save(config, config_path)
    print(f"已生成评估配置：{config_path}", flush=True)
    print(f"评估使用权重：{config.model.pretrained}", flush=True)
    if args.prepare_only:
        return

    # The pinned LAVIS code uses "bert-base-uncased" as a relative local model name.
    print("开始单卡 VQAv2 validation 评估……", flush=True)
    command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=1",
        str(source / "evaluate.py"),
        "--cfg-path", str(config_path),
    ]
    result = subprocess.run(command, cwd=root / "models")
    if result.returncode:
        raise SystemExit(result.returncode)

    print(f"评估完成。结果位于 {config.run.output_dir} 的运行子目录中。", flush=True)
    print("查看 evaluate.txt 中的 agg_metrics（百分制）。", flush=True)
    if not is_subset and not checkpoint:
        print("官方模型在全量验证集的配置参考值约 63.1。", flush=True)


if __name__ == "__main__":
    main()
