"""Offline BLIP-2 VQA models with an optional question-conditioned channel gate.

Torch and LAVIS imports are lazy: data preparation and --help do not need Torch.
The original pinned LAVIS source and the earlier experiment are never modified.
"""

import gc
import os
from pathlib import Path
import sys


COMMIT = "8491f1d333cce678b1230cb940fb249253f3ea4d"
KIND = "blip2_question_channel_gate_v1"
OLD_STAGE2_KIND = "blip2_local_stage2_v1"
VARIANTS = ("plain", "constant", "question")
PROMPT = "Question: {} Short answer:"
_GATE_CLASS = None
_MODEL_CLASS = None


def configure_runtime(root):
    """Configure the uploaded offline model files before importing Transformers."""
    root = Path(root).expanduser().resolve()
    paths = {
        "root": root,
        "source": root / f"LAVIS-{COMMIT}",
        "models": root / "models",
        "t5": root / "models/flan-t5-xl",
        "bert": root / "models/bert-base-uncased",
        "eva": root / "torch-cache/hub/checkpoints/eva_vit_g.pth",
    }
    required = [paths["source"] / "lavis/models/blip2_models/blip2_t5.py",
                paths["eva"], paths["bert"] / "config.json",
                paths["bert"] / "vocab.txt", paths["bert"] / "pytorch_model.bin",
                paths["t5"] / "config.json", paths["t5"] / "tokenizer_config.json",
                paths["t5"] / "tokenizer.json", paths["t5"] / "spiece.model",
                paths["t5"] / "pytorch_model.bin.index.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少离线运行文件：\n" + "\n".join(missing))
    import json
    index = json.loads((paths["t5"] / "pytorch_model.bin.index.json").read_text(encoding="utf-8"))
    missing = [str(paths["t5"] / name) for name in set(index["weight_map"].values())
               if not (paths["t5"] / name).is_file()]
    if missing:
        raise FileNotFoundError("缺少 T5 权重分片：\n" + "\n".join(missing))
    os.environ.update(HF_HOME=str(root / "huggingface"),
                      TORCH_HOME=str(root / "torch-cache"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    source_string = str(paths["source"])
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    # This revision hardcodes the relative BERT name in its Q-Former constructor.
    os.chdir(paths["models"])
    return paths


def default_stage2_checkpoint(root):
    root = Path(root).expanduser().resolve()
    for path in (root / "output/stage2_local_5000/checkpoint_last.pth",
                 root / "checkpoints/stage2/stage2_model_only.pth",
                 root / "final_backup/checkpoints/stage2/stage2_model_only.pth"):
        if path.is_file():
            return path
    candidates = list((root / "compact_weights").glob("*/stage2_model_only.pth"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError("找不到原第二阶段权重，请使用 --init-checkpoint 指定其绝对路径。")


def _gate_class():
    global _GATE_CLASS
    if _GATE_CLASS is not None:
        return _GATE_CLASS
    import torch
    from torch import nn

    class QuestionChannelGate(nn.Module):
        """181056 parameters at dimensions 2048 -> 64 -> 768.

        The up projection starts at zero, so the initial gate is exactly one.
        The down projection keeps its standard random initialization; it is not
        zeroed, which lets the up projection receive a gradient on the first step.
        """

        def __init__(self, question_dim=2048, visual_dim=768, hidden_dim=64, rho=0.25):
            super().__init__()
            if min(question_dim, visual_dim, hidden_dim) < 1 or not 0 < rho < 1:
                raise ValueError("Gate dimensions must be positive and rho must be in (0, 1)")
            self.question_norm = nn.LayerNorm(question_dim, elementwise_affine=False)
            self.down = nn.Linear(question_dim, hidden_dim)
            self.activation = nn.GELU()
            self.up = nn.Linear(hidden_dim, visual_dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
            self.rho = float(rho)

        def forward(self, visual_queries, question_embedding):
            hidden = self.activation(self.down(self.question_norm(question_embedding.float())))
            scale = 1.0 + self.rho * torch.tanh(self.up(hidden))
            return visual_queries * scale.to(visual_queries.dtype).unsqueeze(1)

    _GATE_CLASS = QuestionChannelGate
    return _GATE_CLASS


def _model_class():
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import torch
    from lavis.models.blip2_models.blip2_t5 import Blip2T5
    from lavis.processors.blip_processors import BlipQuestionProcessor

    Gate = _gate_class()

    class QuestionGatedBlip2T5(Blip2T5):
        """The same visual/LLM path for train and evaluation in all variants."""

        def __init__(self, t5_model, variant="plain", hidden_dim=64, rho=0.25,
                     max_question_len=64, max_answer_len=16):
            if variant not in VARIANTS:
                raise ValueError(f"Unknown variant {variant}; choose one of {VARIANTS}")
            super().__init__(img_size=224, num_query_token=32, t5_model=str(t5_model),
                             freeze_vit=True, max_txt_len=max_question_len,
                             apply_lemmatizer=False)
            self.variant = variant
            self.max_question_len = int(max_question_len)
            self.max_answer_len = int(max_answer_len)
            self.question_processor = BlipQuestionProcessor(max_words=50)
            self.channel_gate = (None if variant == "plain" else
                                 Gate(question_dim=self.t5_model.config.hidden_size,
                                      visual_dim=self.Qformer.config.hidden_size,
                                      hidden_dim=int(hidden_dim), rho=float(rho)))
            self.improvement_config = dict(variant=variant, hidden_dim=int(hidden_dim), rho=float(rho),
                                           max_question_len=self.max_question_len,
                                           max_answer_len=self.max_answer_len, prompt=PROMPT)

        def _questions(self, questions, batch_size):
            if isinstance(questions, str):
                questions = [questions] * batch_size
            questions = list(questions)
            if len(questions) != batch_size:
                raise ValueError("Question count does not match image batch size")
            return [self.question_processor(str(question)) for question in questions]

        def _question_embedding(self, questions, device):
            if self.variant == "constant":
                return torch.zeros(len(questions), self.t5_model.config.hidden_size,
                                   device=device, dtype=torch.float32)
            # The question representation is a bag of frozen token embeddings,
            # not an additional contextual T5 encoder pass. No prompt tokens.
            with torch.no_grad():
                tokens = self.t5_tokenizer(questions, padding="longest", truncation=True,
                                          max_length=self.max_question_len,
                                          return_tensors="pt").to(device)
                mask = tokens.attention_mask.bool()
                for special_id in (self.t5_tokenizer.pad_token_id, self.t5_tokenizer.eos_token_id):
                    if special_id is not None:
                        mask = mask & tokens.input_ids.ne(special_id)
                embeddings = self.t5_model.encoder.embed_tokens(tokens.input_ids).float()
                weights = mask.unsqueeze(-1).to(embeddings.dtype)
                return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        def _encoder_inputs(self, images, questions):
            questions = self._questions(questions, images.shape[0])
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(images))
            # predict_answers in the pinned model also restores this to FP32.
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=images.device)
            query = self.query_tokens.expand(images.shape[0], -1, -1)
            query_output = self.Qformer.bert(query_embeds=query,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True)
            queries = query_output.last_hidden_state
            if self.channel_gate is not None:
                questions_embed = self._question_embedding(questions, images.device)
                queries = self.channel_gate(queries, questions_embed)
            visual_tokens = self.t5_proj(queries)
            visual_atts = torch.ones(visual_tokens.size()[:-1], dtype=torch.long, device=images.device)
            tokens = self.t5_tokenizer([PROMPT.format(question) for question in questions],
                                      padding="longest", truncation=True,
                                      max_length=self.max_question_len,
                                      return_tensors="pt").to(images.device)
            with self.maybe_autocast(dtype=torch.bfloat16):
                text_embeds = self.t5_model.encoder.embed_tokens(tokens.input_ids)
                inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            attention_mask = torch.cat([visual_atts, tokens.attention_mask], dim=1)
            return inputs_embeds, attention_mask

        def forward_qa(self, samples):
            images = samples["image"]
            questions = samples.get("question", samples.get("text_input"))
            answers = samples.get("answer", samples.get("text_output"))
            if questions is None or answers is None:
                raise ValueError("forward_qa needs image, question and answer")
            if isinstance(answers, str):
                answers = [answers]
            if len(answers) != images.shape[0]:
                raise ValueError("Answer count does not match image batch size")
            inputs_embeds, attention_mask = self._encoder_inputs(images, questions)
            targets = self.t5_tokenizer(list(answers), padding="longest", truncation=True,
                                       max_length=self.max_answer_len,
                                       return_tensors="pt").to(images.device)
            labels = targets.input_ids.masked_fill(
                targets.input_ids == self.t5_tokenizer.pad_token_id, -100)
            with self.maybe_autocast(dtype=torch.bfloat16):
                outputs = self.t5_model(inputs_embeds=inputs_embeds,
                                        attention_mask=attention_mask,
                                        decoder_attention_mask=targets.attention_mask,
                                        labels=labels, return_dict=True)
            return {"loss": outputs.loss}

        def forward(self, samples):
            return self.forward_qa(samples)

        @torch.no_grad()
        def answer_questions(self, images, questions, num_beams=5, max_new_tokens=10,
                             min_length=1, length_penalty=-1.0):
            inputs_embeds, attention_mask = self._encoder_inputs(images, questions)
            with self.maybe_autocast(dtype=torch.bfloat16):
                outputs = self.t5_model.generate(inputs_embeds=inputs_embeds,
                                                 attention_mask=attention_mask,
                                                 do_sample=False, num_beams=num_beams,
                                                 max_new_tokens=max_new_tokens,
                                                 min_length=min_length,
                                                 length_penalty=length_penalty)
            return self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        @torch.no_grad()
        def predict_answers(self, samples, num_beams=5, max_len=10, min_len=1,
                            length_penalty=-1.0, **kwargs):
            questions = samples.get("question", samples.get("text_input"))
            if questions is None:
                raise ValueError("predict_answers needs raw questions in question/text_input")
            return self.answer_questions(samples["image"], questions, num_beams=num_beams,
                                         max_new_tokens=max_len, min_length=min_len,
                                         length_penalty=length_penalty)

        @torch.no_grad()
        def generate(self, samples, num_beams=5, max_length=10, min_length=1,
                     length_penalty=-1.0, **kwargs):
            questions = samples.get("question", samples.get("text_input"))
            if questions is None:
                raise ValueError("This VQA model requires raw samples['question']; use answer_questions")
            return self.answer_questions(samples["image"], questions, num_beams=num_beams,
                                         max_new_tokens=max_length, min_length=min_length,
                                         length_penalty=length_penalty)

    _MODEL_CLASS = QuestionGatedBlip2T5
    return _MODEL_CLASS


def __getattr__(name):
    if name == "QuestionChannelGate":
        return _gate_class()
    if name == "QuestionGatedBlip2T5":
        return _model_class()
    raise AttributeError(name)


def _validate_state(state, shapes, description):
    if not isinstance(state, dict):
        raise ValueError(f"{description}: model is not a parameter dictionary")
    missing = sorted(set(shapes) - set(state))
    extra = sorted(set(state) - set(shapes))
    wrong = [name for name in shapes if name in state and tuple(state[name].shape) != tuple(shapes[name])]
    if missing or extra or wrong:
        raise ValueError(f"{description}: missing={missing[:6]}, extra={extra[:6]}, shape={wrong[:6]}")


def setup_model(root, variant="plain", init_checkpoint=None, device="cuda",
                hidden_dim=64, rho=0.25, max_question_len=64, max_answer_len=16):
    """Initialize every baseline from exactly the same user's stage-2 weights.

    Model construction consumes RNG. Training should seed its DataLoader and
    answer sampler independently, or reseed them after this call, so adding a
    gate does not change the common data sequence between variants.
    """
    root = Path(root).expanduser().resolve()
    checkpoint_path = (Path(init_checkpoint).expanduser().resolve() if init_checkpoint
                       else default_stage2_checkpoint(root))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    paths = configure_runtime(root)
    import torch
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("kind") != OLD_STAGE2_KIND:
        raise ValueError("Initialization must be your old stage-2 checkpoint, not the official adapter or an improved run")
    state = payload["model"]
    del payload  # Release old optimizer/RNG state before allocating the model.
    gc.collect()
    model = _model_class()(paths["t5"], variant=variant, hidden_dim=hidden_dim, rho=rho,
                            max_question_len=max_question_len, max_answer_len=max_answer_len)
    shapes = {name: parameter.shape for name, parameter in model.named_parameters()
              if parameter.requires_grad and not name.startswith("channel_gate.")}
    _validate_state(state, shapes, "Original stage-2 weights")
    model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    model.initialization_checkpoint = str(checkpoint_path)
    return model.to(device)


def trainables_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def checkpoint_payload(model, **metadata):
    reserved = {"kind", "model", "model_config", "source_commit"}
    if reserved.intersection(metadata):
        raise ValueError(f"Reserved checkpoint metadata keys: {sorted(reserved.intersection(metadata))}")
    return dict(kind=KIND, source_commit=COMMIT, model_config=dict(model.improvement_config),
                model=trainables_state(model), **metadata)


def load_improved_model(root, path, device="cuda"):
    """Reconstruct the saved architecture before strict trainable-key loading."""
    path = Path(path).expanduser().resolve()
    paths = configure_runtime(root)
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != KIND or payload.get("source_commit") != COMMIT:
        raise ValueError("Not a checkpoint from this improvement experiment/version")
    config = dict(payload["model_config"])
    if config.pop("prompt", PROMPT) != PROMPT:
        raise ValueError("Checkpoint VQA prompt differs from this implementation")
    state = payload["model"]
    metadata = {key: value for key, value in payload.items()
                if key not in {"model", "optimizer", "python_rng", "torch_rng", "cuda_rng", "loader_rng"}}
    del payload
    gc.collect()
    model = _model_class()(paths["t5"], **config)
    _validate_state(state, {name: parameter.shape for name, parameter in model.named_parameters()
                            if parameter.requires_grad}, "Improved model weights")
    model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    return model.to(device).eval(), metadata
