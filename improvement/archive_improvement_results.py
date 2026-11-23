"""Archive completed VQA ablations and write an honest Markdown report. CPU only."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from analyze_improvement_results import read_run, paired_interval, VARIANTS


SCREENSHOT_SCORES = {
    "dev": {
        "plain": {"agg_metrics": 41.57, "number": 25.22, "other": 23.94, "yes/no": 68.82},
        "constant": {"agg_metrics": 42.32, "number": 27.72, "other": 24.11, "yes/no": 69.67},
        "question": {"agg_metrics": 41.29, "number": 27.28, "other": 24.15, "yes/no": 67.12}},
    "test": {
        "plain": {"agg_metrics": 39.84, "number": 18.58, "other": 25.19, "yes/no": 68.23},
        "constant": {"agg_metrics": 40.09, "number": 18.51, "other": 25.45, "yes/no": 68.59},
        "question": {"agg_metrics": 39.51, "number": 18.66, "other": 24.91, "yes/no": 67.67}},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def make_report(scores, comparisons=None, facts=None):
    facts = facts or {}
    comparisons = comparisons or {}
    dev, test = scores["dev"], scores["test"]
    overall = "\n".join(f"| {v} | {dev[v]['agg_metrics']:.2f} | {test[v]['agg_metrics']:.2f} | "
                         f"{test[v]['agg_metrics']-test['plain']['agg_metrics']:+.2f} |" for v in VARIANTS)
    categories = "\n".join(f"| {v} | {test[v]['number']:.2f} | {test[v]['other']:.2f} | "
                            f"{test[v]['yes/no']:.2f} |" for v in VARIANTS)
    uncertainty = "尚未在此稿中计算配对置信区间，不将 0.25 个百分点称为显著或稳定提升。"
    if comparisons:
        rows = "\n".join(f"| {name} | {r['difference_pp']:+.3f} | "
                         f"[{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}] |"
                         for name, r in comparisons.items())
        uncertainty = ("按图片成组、对模型间差值进行 2000 次配对 bootstrap；每次抽中图片后保留其全部问题。\n\n"
                       "| 比较（前者减后者，百分点） | 差值 | 95% percentile 区间 |\n"
                       "|---|---:|---:|\n" + rows + "\n\n"
                       "区间若包含 0，当前图片抽样下不足以支持明确方向的收益；即使不包含 0，"
                       "也仅反映图片抽样的不确定性，不能替代多训练随机种子或跨数据集实验。")
    return f"""# BLIP-2 通道门控改进尝试及消融实验报告


## 摘要

本实验在已有 BLIP-2 小规模两阶段训练结果上，研究轻量通道门控对视觉问答的影响。保持 EVA 和 Flan-T5-XL 冻结，设置普通 VQA 微调、常量条件门控和问题条件门控三组同预算实验。每组使用 10000 道训练题，微调 1000 步，并分别在开发集和独立留出的测试集上评估。测试集总体准确率分别为 {test['plain']['agg_metrics']:.2f}%、{test['constant']['agg_metrics']:.2f}% 和 {test['question']['agg_metrics']:.2f}%。常量门控呈现微小正向变化，问题条件门控未优于基线；结果不支持将新增问题分支表述为已验证有效的改进。

## 1. 背景与实验假设

BLIP-2 使用 Q-Former 连接视觉编码器与语言模型。本地已完成两阶段各 5000 步的小规模训练。本实验从该第二阶段检查点继续进行监督 VQA 微调，考察“对现有视觉表示进行轻量通道调制，能否改善小样本问答”。

原论文在 VQA 微调中已将问题输入 Q-Former，InstructBLIP 和 FiLM 等工作也涉及条件化特征调制。因此本实验定位为已有思想在现有管线上的低预算适配与实证，不声称首次提出问题条件或门控机制。

## 2. 方法

Q-Former 输出 32 个 768 维向量。问题条件分支复用冻结 T5 的词嵌入，对排除 PAD/EOS 的原始问题 token 取均值，得到 2048 维表示 q；它不是额外的上下文文本编码器。

门控采用无仿射参数 LayerNorm 和 2048→64→768 的两层 MLP：

$$g(q)=1+0.25\\tanh(W_{{up}}\\operatorname{{GELU}}(W_{{down}}\\operatorname{{LN}}(q))).$$

$$Z'=Z\\odot g(q).$$

同一通道向量广播到 32 个 query，逐通道缩放在 0.75–1.25 内；不增加或裁剪 query。上投影权重与偏置置零，使初始门控为恒等；下投影正常随机初始化。新增参数共 181056 个。

| 组别 | 配置 | 作用 |
|---|---|---|
| plain | 无门控，普通监督 VQA 微调 | 主基线 |
| constant | 相同门控，输入全零常量问题表示 | 控制额外参数和全局通道校准因素 |
| question | 使用真实问题的冻结词嵌入均值 | 检验问题条件分支 |

三组共同训练 Q-Former、query、视觉 LayerNorm 和原 T5 投影层；后两组额外训练门控。constant 的 18 万参数并不代表必须用这么大的模块才能表达一个固定通道向量；它在这里用于控制模块大小。

## 3. 数据与设置

| 项目 | 设置 |
|---|---|
| 硬件 | 单张 NVIDIA A800-SXM4-80GB |
| 初始化 | 同一个自训练 BLIP-2 第二阶段权重 |
| 训练集 | 原 caption 训练图片对应的 10000 道 VQA，每图最多一道 |
| 开发集 | 原先已查看结果的 1000 道题，931 张图 |
| 最终测试集 | 另外留出的 1000 道题，923 张图 |
| 图像 | 224×224，固定 resize 和标准化，不随机裁剪或翻转 |
| 步数／batch／种子 | 每组 1000／16／42 |
| 学习率 | 公共模块 1e-5，门控 1e-4 |
| 优化器 | AdamW，beta=(0.9,0.98)，weight decay=0.05 |
| 调度 | 50 步 warmup，随后逐步余弦衰减到峰值的 0.1 |
| 精度／梯度裁剪 | BF16／最大梯度范数 1 |
| 答案采样 | 从原始 10 个答案中均匀抽取一票，保留答案频率 |
| 生成与评分 | 5-beam，max_new_tokens=10，固定版本官方 VQA 软准确率 |

训练、开发与测试图片互不重叠。所有新增实验使用相同样本顺序、答案抽样和公共训练预算。QA 微调后属于监督实验，不称为 zero-shot。本地划分基于现有 VQAv2 validation 图片，不是官方 train/test benchmark 划分。图片隔离仅针对本次本地图文训练，不保证公开基础模型从未见过这些图片。

## 4. 结果

| 模型 | 开发集（%） | 测试集（%） | 测试集相对 plain（百分点） |
|---|---:|---:|---:|
{overall}

| 模型 | 测试 number（%） | 测试 other（%） | 测试 yes/no（%） |
|---|---:|---:|---:|
{categories}

常量门控相对基线的开发集变化为 {dev['constant']['agg_metrics']-dev['plain']['agg_metrics']:+.2f} 个百分点，测试集变化为 {test['constant']['agg_metrics']-test['plain']['agg_metrics']:+.2f} 个百分点。问题条件门控相对基线的测试变化为 {test['question']['agg_metrics']-test['plain']['agg_metrics']:+.2f} 个百分点。

{uncertainty}

分数来源：{facts.get('score_source', '开发集与测试集结果截图；此稿未直接读取服务器结果')}。

## 5. 分析与局限

当前结果未支持“加入真实问题条件必然优于简单通道校准”的假设。冻结词嵌入均值丢失词序，后置门控无法补回原视觉表示已丢失的信息，有限数据及优化设置也可能影响效果；这些是可能解释，现有实验未逐一证明。

不能把普通 QA 微调相对旧 caption-only 模型的变化全部归因于门控。旧的 41.90% 和官方 65.48% 是原开发题集上的历史参照，不能拿来与这里的新测试集成绩直接比较。开发与测试题集不同，分数下降本身也不能单独证明过拟合。

实验只有一个训练种子和一个小规模划分，效应很小，不声称统计显著、稳定提升、最优结构或达到论文精度。最终测试后不继续根据测试成绩调整参数。

## 6. 结论

本实验完成了 BLIP-2 轻量通道门控的实现、同预算基线与消融对照。常量门控在单次实验中表现出微小正向变化，问题条件分支未体现额外收益。这一结果支持如实报告改进尝试及其边界，而不将未获支持的假设包装为成功的精度改进。

## 7. 成果保存

保存三组最终精简检查点、训练配置及样本计划、训练日志、六份开发／测试评估、逐题预测及得分、数据划分、代码和配对统计。检查点单独下载，证据 ZIP 不重复包含大型权重。恢复模型仍需要原来已保存的 EVA、BERT、Flan-T5-XL 及原始两阶段结果。

## 参考资料

- [BLIP-2 原论文](https://proceedings.mlr.press/v202/li23q/li23q.pdf)
- [FiLM](https://arxiv.org/abs/1709.07871)
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
"""


VERIFY = '''import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST_improvement.json').read_text(encoding='utf-8'))
failed=[]
for entry in manifest['files']:
    path=(root/entry['local_path']).resolve()
    if not path.is_relative_to(root):
        raise SystemExit('Unsafe manifest path')
    if not path.is_file() or path.stat().st_size!=entry['size']:
        print('MISSING OR SIZE MISMATCH:',entry['local_path']);failed.append(entry['local_path']);continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    if h.hexdigest()!=entry['sha256']:
        print('HASH MISMATCH:',entry['local_path']);failed.append(entry['local_path'])
if failed:raise SystemExit(f'Verification failed: {len(failed)} files')
print('ALL IMPROVEMENT FILES VERIFIED: evidence and three model checkpoints.')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    base = root / "output/improvement"
    runs = {split: {v: read_run(base, v, split, 42, 1000) for v in VARIANTS} for split in ("dev", "test")}
    first = runs["test"]["plain"]["metadata"]["checkpoint_metadata"]
    for split in runs:
        reference = runs[split]["plain"]["metadata"]
        for v, run in runs[split].items():
            meta = run["metadata"]
            ckpt = meta["checkpoint_metadata"]
            if ckpt["step"] != 1000 or ckpt["step"] != ckpt["total_steps"]:
                raise SystemExit("归档要求三组均完成 1000 步。")
            if meta["split_sha256"] != reference["split_sha256"] or meta["decoding"] != reference["decoding"]:
                raise SystemExit("评估题集或解码不一致，请核对，不自动生成同预算结论。")
            common = ("seed", "total_steps", "batch_size", "base_lr", "train_sha256", "sample_plan_sha256", "split_manifest_sha256")
            if ckpt["init_checkpoint_sha256"] != first["init_checkpoint_sha256"] or any(
                    ckpt["run_config"][k] != first["run_config"][k] for k in common):
                raise SystemExit("初始化、数据计划或训练预算不一致，请核对。")
            if ckpt["model_config"]["variant"] != v:
                raise SystemExit("模型 variant 与评估目录不一致。")
            cfg = ckpt["run_config"]
            if any(cfg.get(k) != value for k, value in
                   {"batch_size": 16, "seed": 42, "base_lr": 1e-5, "gate_lr": 1e-4,
                    "hidden_dim": 64, "rho": 0.25, "warmup_steps": 50}.items()):
                raise SystemExit("实际超参数与本轮报告设置不同，请核对后再生成报告。")
            if any(abs(run["metrics"][k] - value) > 0.005
                   for k, value in SCREENSHOT_SCORES[split][v].items()):
                raise SystemExit("最新评估与本轮已提供分数不同，请核对目录，避免套用不匹配的结论。")
    print("CPU 配对统计：读取已有逐题结果，不训练、不重新评估……", flush=True)
    comparisons = {}
    for a, b in (("plain", "constant"), ("plain", "question"), ("constant", "question")):
        comparisons[f"{b}-{a}"] = paired_interval(runs["test"][a]["questions"], runs["test"][b]["questions"])
    scores = {s: {v: run["metrics"] for v, run in variants.items()} for s, variants in runs.items()}
    sources = set()
    for path in (root / "improvement").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".json", ".txt"}:
            sources.add(path)
    for path in (root / "data/coco/improvement_v1").glob("*.json"):
        sources.add(path)
    for v in VARIANTS:
        run_folder = base / f"{v}_seed42_steps1000"
        for name in ("run_config.json", "sample_plan.json", "train_log.jsonl"):
            path = run_folder / name
            if not path.is_file():
                raise SystemExit(f"缺少需要保存的训练证据：{path}")
            sources.add(path)
        for s in ("dev", "test"):
            sources.update(Path(runs[s][v]["path"]).glob("*.json"))
    entries = []
    for v in VARIANTS:
        path = base / f"{v}_seed42_steps1000/checkpoint_last.pth"
        if not path.is_file():
            raise SystemExit(f"缺少最终模型：{path}")
        print(f"核对 {v} 检查点 SHA256（{path.stat().st_size / 1024**2:.1f} MiB）……", flush=True)
        digest = sha256(path)
        if any(digest != runs[s][v]["metadata"]["checkpoint_sha256"] for s in ("dev", "test")):
            raise SystemExit(f"{v} 当前权重与已评估权重不同，请核对。")
        entries.append(dict(local_path=f"checkpoints/{v}/checkpoint_last.pth", server_path=str(path),
                            size=path.stat().st_size, sha256=digest, included_in_zip=False))
    for path in sorted(sources):
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 256 * 1024**2:
            raise SystemExit(f"证据路径不安全或过大：{path}")
        entries.append(dict(local_path="project/" + path.relative_to(root).as_posix(), server_path=str(path),
                            size=path.stat().st_size, sha256=sha256(path), included_in_zip=True))
    if shutil.disk_usage(root).free < sum(e["size"] for e in entries if e["included_in_zip"]) + 20 * 1024**2:
        raise SystemExit("小文件证据包空间不足；未修改原始结果。")
    generated = {
        "BLIP-2_通道门控改进实验报告.md": make_report(scores, comparisons, {"score_source": "服务器原始 metrics.json 和逐题得分，已核对运行元信息"}),
        "comparison.json": json.dumps(dict(scores=scores, comparisons=comparisons), ensure_ascii=False, indent=2),
        "verify_improvement_backup.py": VERIFY,
        "README_保存说明.txt": "将三个模型分别放入 checkpoints/plain、checkpoints/constant、checkpoints/question，文件名均为 checkpoint_last.pth。\n运行 python verify_improvement_backup.py 校验。ZIP不含模型权重。报告正文保留负结果和统计边界。\n",
    }
    for name, text in generated.items():
        content = text.encode("utf-8")
        entries.append(dict(local_path=name, size=len(content), sha256=hashlib.sha256(content).hexdigest(), included_in_zip=True))
    target = root / "final_archive"
    target.mkdir(exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    archive = target / f"blip2_improvement_results_{stamp}.zip"
    temp = archive.with_suffix(".zip.part")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for entry in entries:
            if entry["included_in_zip"] and "server_path" in entry:
                z.write(entry["server_path"], entry["local_path"])
        for name, text in generated.items():
            z.writestr(name, text)
        z.writestr("MANIFEST_improvement.json", json.dumps(dict(files=entries), ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise SystemExit("ZIP 校验失败；原始文件保留。")
    temp.replace(archive)
    print(f"\n证据 ZIP：{archive}\n大小：{archive.stat().st_size / 1024**2:.1f} MiB", flush=True)
    print("还需分别下载三个精简模型，放到不同的本地子文件夹：", flush=True)
    for entry in entries[:3]:
        print(f"{entry['server_path']}\n  -> {entry['local_path']}", flush=True)
    print("解压 ZIP 后运行 python verify_improvement_backup.py。原实验文件没有修改。")


if __name__ == "__main__":
    main()
