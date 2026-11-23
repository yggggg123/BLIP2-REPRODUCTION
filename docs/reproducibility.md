# Reproducibility and local file layout

Scripts resolve the project root from their own location. Put the downloaded dependencies at the locations below; the large files are deliberately not committed.

```text
PROJECT_ROOT/
  LAVIS-8491f1d333cce678b1230cb940fb249253f3ea4d/
  models/bert-base-uncased/      config, tokenizer files, pytorch_model.bin
  models/flan-t5-xl/            config, tokenizer files, .bin index and both shards
  torch-cache/hub/checkpoints/
    eva_vit_g.pth
    blip2_pretrained_flant5xl.pth   # only for official-model reference evaluation
  data/coco/images/val2014/     original COCO JPEGs
  data/coco/annotations/        VQAv2 inputs and original ten-answer annotations
```

Model sources: [BERT](https://huggingface.co/google-bert/bert-base-uncased), [Flan-T5-XL](https://huggingface.co/google/flan-t5-xl), and the EVA/BLIP-2 URLs in the pinned LAVIS files `lavis/models/eva_vit.py` and `lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml`. Model and dataset files remain subject to their providers' terms.

COCO validation images: `http://images.cocodataset.org/zips/val2014.zip`. VQAv2 question/answer files are available from [the official dataset site](https://visualqa.org/download.html). The original evaluation helper also retrieves the four expected JSON files from LAVIS storage when they are missing.

## Caption training subset

Download the original Karpathy train annotation JSON from:

```text
https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
```

Then run:

```bash
python scripts/prepare_caption_subset.py --input /path/to/coco_karpathy_train.json
```

The experiment's expected result is 152634 caption pairs / 30504 images. The expected serialized file SHA256 is `4aefcc81b60c316ac0721251724c651ab4df90cfb346e4167d7542716be97d09`. Different source contents/order can produce different hashes, even if counts agree.

## Original development input

After restoring your trained stage-2 checkpoint and VQA data, reconstruct the original 1000-question selection without running a GPU evaluation:

```bash
python run_vqav2_local.py \
  --checkpoint /path/to/stage2_model_only.pth --limit 1000 --seed 42 --prepare-only
```

This creates `data/coco/annotations/holdout_n1000_seed42/vqa_val_eval.json`. The following helper creates the new training/development/test files:

```bash
python improvement/prepare_vqa_improvement.py
```

Compare the resulting IDs with `splits/*_ids.json`. The committed `splits/split_manifest.json` is an archived reference, not a substitute for generating the actual local annotations and checking their hashes. Ten human-answer votes per question are used for sampling/scoring; no full ground-truth text is redistributed in this repository.

## Initialization and reproducibility limits

All ablations must initialize from exactly the same self-trained stage-2 parameters. Check their recorded initialization hash and sample-plan hash. New training checks the hashes of the locally generated split files. Final model-only checkpoints cannot restore optimizer/RNG state; an incomplete run with optimizer state can be resumed within its original planned budget.

The first-stage `load_pretrained` factory repair is applied by `train_blip2_stage1_local.py` on the pinned source and is backed up there. Other LAVIS versions are not supported by that text repair. Do not substitute current main or a different PyPI LAVIS version and expect the same behavior.

Different hardware, kernels, library versions, or training randomness can alter numbers. This study has one training seed. Paired image bootstrap measures evaluation-image sampling uncertainty, not full training reproducibility. The original two-stage schedule and the later custom VQA schedule are documented separately in the reports.

`configs/environment_observed.txt` omits the original absolute installation prefix, using a relative LAVIS editable path instead. It records the used environment; it is not proof that every package combination can be newly installed today.
