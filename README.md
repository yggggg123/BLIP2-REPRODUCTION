# BLIP-2 Small-Scale Reproduction and Channel-Gating Study

单张 A800 80GB 上的 BLIP-2 小规模两阶段复现，以及轻量通道门控的监督 VQA 消融实验。项目用于小规模复现与方法验证，保留实际结果及未获支持的假设，不宣称达到原论文训练规模或精度。

## 结果

原两阶段训练各 5000 步，在原先的 1000 道问题上，自训练模型为 **41.90%**，官方模型为 **65.48%**。该问题集在后续门控实验中作为开发集，不能与新的测试集直接比较。

后续三组均从相同的自训练第二阶段权重开始，在相同 10000 道 QA 上微调 1000 步：

| 模型 | 开发集 VQA（%） | 新测试集 VQA（%） | 新测试集相对基线（百分点） |
|---|---:|---:|---:|
| plain：普通微调 | 41.57 | 39.84 | — |
| constant：常量条件门控 | 42.32 | 40.09 | +0.25 |
| question：问题条件门控 | 41.29 | 39.51 | −0.33 |

常量门控相对基线的图片配对 bootstrap 95% 区间约为 **[−0.35, +0.86] 个百分点**，包含 0。当前单训练种子、小测试集的结果不足以支持明确的性能提升；问题条件分支也未显示额外收益。

- [原始复现报告](docs/reproduction_report.md)
- [改进与消融报告](docs/gating_report.md)
- [论文精讲笔记](docs/paper_notes.md)
- [机器可读结果与配对区间](results/comparison.json)

## 方法和实验范围

基础模型为冻结的 EVA ViT-g 与 Flan-T5-XL。前两阶段训练 Q-Former、query、视觉 LayerNorm 和相应任务头／投影层。

门控实验在 Q-Former 输出后加入 `2048 → 64 → 768` MLP，共 **181056** 个新增参数：

```text
image → EVA → Q-Former → channel gate → T5 projection → Flan-T5 → answer
                            ↑
question → frozen T5 token-embedding mean → small MLP
```

门控为 `1 + 0.25*tanh(MLP(LayerNorm(q)))`，上投影零初始化、初始恒等，范围 0.75–1.25。constant 使用全零条件；question 使用原问题的冻结词嵌入均值。它属于 FiLM 类思路的轻量适配，不声称首次提出问题条件机制。均值表示丢失词序，后置调制不能补回已经丢失的视觉细节。

## 目录

```text
improvement/             三组模型、训练、评估、配对统计与归档代码
scripts/                 从官方 caption 标注重建本地训练子集
configs/                 三组实际运行配置及参考环境清单
splits/                  原划分清单及 question_id / image_id；不含答案正文
results/                 原始指标、预测、逐题得分及训练日志
docs/                    报告、讲义和复现说明
licenses/                第三方 LAVIS 许可证
train_blip2_stage*.py     原小规模两阶段训练入口
run_*_local.py           原推理和 VQAv2 评估入口
```

权重、COCO 图片、数据集标注正文、虚拟环境、缓存、备份 ZIP 和整份第三方源码不放在 Git 仓库。模型检查点的大小和 SHA256 见 [清单](results/checkpoint_inventory.json)，目前未提供公开下载链接。

## 环境

运行环境为 **Linux + NVIDIA GPU**；Windows 可用于查看文件、运行标准库测试和管理 Git，不能直接复用服务器的 Linux venv。

实验使用 Python 3.10.22、PyTorch 2.3.0+cu118、torchvision 0.18.0+cu118、Transformers 4.28.1。依赖快照见 [configs/environment_observed.txt](configs/environment_observed.txt)，它是观测记录，不是已经验证过的全新环境锁文件。

在新的 Linux 环境中，先创建 Python 3.10 环境并安装匹配的 PyTorch：

```bash
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
```

取得固定版本 LAVIS，解压后目录名称必须为 `LAVIS-8491f1d333cce678b1230cb940fb249253f3ea4d`：

```bash
curl -L https://github.com/salesforce/LAVIS/archive/8491f1d333cce678b1230cb940fb249253f3ea4d.zip -o lavis-source.zip
python -m zipfile -e lavis-source.zip .
python -m pip install -c constraints.txt -r requirements.txt
python -m pip install -c constraints.txt -e ./LAVIS-8491f1d333cce678b1230cb940fb249253f3ea4d
```

上述是基于已用环境整理的安装路径，未在全新机器上重新验证安装。已有实验环境无需重装。

## 数据与权重准备

本仓库提供脚本和结果证据，不包含运行所需的全部大文件。模型、数据路径和重建原开发集的方法见 [docs/reproducibility.md](docs/reproducibility.md)。请先准备它们，不能只 clone 本仓库就跳过模型和数据准备。

## 原小规模两阶段训练

在项目根目录运行。第一阶段脚本会为固定版本 LAVIS 修正 `load_pretrained=False` 开关，并保留源码备份。

```bash
python train_blip2_stage1_local.py --steps 5000 --batch-size 32
python train_blip2_stage2_local.py --steps 5000 --batch-size 16
```

完整第二阶段和精简第二阶段参数都可作为后续 VQA 实验的初始化。使用自己的其他初始化时，不应预期精确复现本仓库数值。

## 三组监督 VQA 实验

先重建旧开发题集，随后固定新的三组图片隔离划分。新划分训练 10000 题／10000 图、开发 1000 题／931 图、测试 1000 题／923 图；属于基于现有 VQAv2 validation 图片的自定义实验，不是官方 train/test benchmark。

```bash
python improvement/prepare_vqa_improvement.py
python improvement/test_gate_cpu.py
```

三组均从相同的第二阶段权重出发，执行：

```bash
for variant in plain constant question; do
  python improvement/train_vqa_improvement.py --variant "$variant" --steps 1000 || break
done
```

评估开发集：

```bash
for variant in plain constant question; do
  python improvement/evaluate_vqa_improvement.py \
    --checkpoint "output/improvement/${variant}_seed42_steps1000/checkpoint_last.pth" \
    --split dev || break
done
```

固定配置后使用同样的评估命令改为 `--split test`。测试集不用于调参。首次尝试可先用 `--stop-after 100` 做检查，续训需明确传入对应 `--resume` 文件；最终精简文件不含优化器状态。

三组实际配置在 `configs/`。图像使用固定 resize 和标准化，避免随机裁剪或翻转改变 VQA 标签；样本及答案顺序固定，保证对照预算一致。

## 检查和统计

不依赖 Torch 的协议测试：

```bash
python -m unittest discover -s improvement/tests -v
```

对已公开的逐题结果直接计算配对区间，无需 GPU 或权重：

```bash
python scripts/analyze_published_results.py
```

这些区间描述图片抽样的不确定性，不包含训练随机性，也不证明跨数据集泛化。

## 参考及许可说明

- [BLIP-2 论文](https://proceedings.mlr.press/v202/li23q/li23q.pdf)
- [LAVIS 固定版本](https://github.com/salesforce/LAVIS/tree/8491f1d333cce678b1230cb940fb249253f3ea4d)
- [FiLM](https://arxiv.org/abs/1709.07871)
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
- [VQAv2 数据](https://visualqa.org/download.html)

第三方来源及保留的 BSD-3-Clause 许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目新增代码的许可证尚待发布者选择，未用第三方许可证代替发布者作出许可决定。
