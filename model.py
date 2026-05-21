"""
Qwen-RADDino: Qwen2.5-7B-Instruct with RAD-DINO Vision Encoder.

Architecture:
  Image (224×224×3)
    → RAD-DINO encoder           → [B, 256, 768]
    → MLP Projector (768→3584)   → [B, 256, 3584]     ← NOTE: 3584, not 4096!
    → Concat with text embeds    → [B, 256+T, 3584]
    → Qwen2.5-7B LLM            → text generation

Key differences from LLaVA-RADDino (Vicuna):
  1. LLM hidden size: 3584 (Qwen) vs 4096 (Vicuna)
  2. Projector: 768→3584→3584 (not 768→4096→4096)
  3. Prompt format: ChatML (<|im_start|>user\n...<|im_end|>) vs Vicuna
  4. Tokenizer: Qwen's tokenizer with different special tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

from raddino_encoder import RadDinoEncoder
import config as cfg


class MultimodalProjector(nn.Module):
    """
    2-layer MLP projector: Linear(768→3584) → GELU → Linear(3584→3584).

    Bridges RAD-DINO's 768-dim features to Qwen2.5-7B's 3584-dim embedding space.
    """

    def __init__(
        self,
        vision_hidden_size: int = cfg.RADDINO_HIDDEN_SIZE,
        llm_hidden_size: int = cfg.LLM_HIDDEN_SIZE,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 768] vision features

        Returns:
            [B, N, 3584] projected features for Qwen2.5
        """
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class QwenRaddino(nn.Module):
    """
    Full Qwen-RADDino model.

    Components:
      1. RadDinoEncoder      — frozen vision encoder (768-dim output)
      2. MultimodalProjector — trainable bridge (768 → 3584)
      3. Qwen2.5-7B-Instruct — language model (frozen S1, LoRA S2)

    Prompt construction (ChatML format):
      <|im_start|>system
      You are a radiology AI assistant...<|im_end|>
      <|im_start|>user
      [IMAGE_TOKENS × 256] Generate a detailed radiology report...<|im_end|>
      <|im_start|>assistant
      [REPORT]<|im_end|>
    """

    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()

        # ---- Vision encoder (frozen) ----
        self.vision_encoder = RadDinoEncoder(cfg.RADDINO_MODEL_NAME)

        # ---- Multimodal projector (trainable, randomly initialized) ----
        # NOTE: Projects to 3584 (Qwen), NOT 4096 (Vicuna)
        self.projector = MultimodalProjector(
            vision_hidden_size=cfg.RADDINO_HIDDEN_SIZE,
            llm_hidden_size=cfg.LLM_HIDDEN_SIZE,      # 3584
        ).to(dtype=torch.bfloat16)

        print(f"[QwenRaddino] Projector: {cfg.RADDINO_HIDDEN_SIZE} → {cfg.LLM_HIDDEN_SIZE} → {cfg.LLM_HIDDEN_SIZE}")

        # ---- Language model ----
        print(f"[QwenRaddino] Loading LLM: {cfg.LLM_MODEL_NAME} ...")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            cfg.LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
        self.language_model.resize_token_embeddings(len(tokenizer))
        print(f"[QwenRaddino] LLM loaded. Parameters: {sum(p.numel() for p in self.language_model.parameters()) / 1e6:.1f}M")
        print(f"[QwenRaddino] LLM hidden_size: {self.language_model.config.hidden_size}")

        # Verify dimension match
        assert self.language_model.config.hidden_size == cfg.LLM_HIDDEN_SIZE, (
            f"LLM hidden_size mismatch: config says {cfg.LLM_HIDDEN_SIZE}, "
            f"model has {self.language_model.config.hidden_size}"
        )

        # ---- Pre-tokenize ChatML prompt parts ----
        # System + User prefix (before image tokens)
        self.prefix_text = (
            f"<|im_start|>system\n{cfg.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        # User suffix (after image tokens) + assistant start
        self.suffix_text = (
            f"\n{cfg.USER_PROMPT}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        prefix_tokens = tokenizer(
            self.prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        suffix_tokens = tokenizer(
            self.suffix_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        self.register_buffer("prefix_ids", prefix_tokens.input_ids)
        self.register_buffer("suffix_ids", suffix_tokens.input_ids)

        # Store im_end token id for report termination
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        print(f"[QwenRaddino] Prompt: prefix={self.prefix_ids.shape[1]} tokens, "
              f"suffix={self.suffix_ids.shape[1]} tokens, "
              f"image={cfg.NUM_IMAGE_TOKENS} tokens")

    @property
    def llm_dtype(self) -> torch.dtype:
        """Get the dtype of the LLM weights (bf16)."""
        return next(self.language_model.parameters()).dtype

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images through RAD-DINO + projector.

        Args:
            pixel_values: [B, 3, 224, 224]

        Returns:
            image_embeds: [B, 256, 3584] in LLM dtype (bf16)
        """
        patch_features = self.vision_encoder(pixel_values)  # [B, 256, 768]

        patch_features = patch_features.to(dtype=self.projector.linear_1.weight.dtype)
        image_embeds = self.projector(patch_features)  # [B, 256, 3584]

        image_embeds = image_embeds.to(dtype=self.llm_dtype)

        B = pixel_values.shape[0]
        assert image_embeds.shape == (B, cfg.NUM_IMAGE_TOKENS, cfg.LLM_HIDDEN_SIZE), (
            f"Projector output mismatch: expected ({B}, {cfg.NUM_IMAGE_TOKENS}, "
            f"{cfg.LLM_HIDDEN_SIZE}), got {image_embeds.shape}"
        )

        return image_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        report_ids: torch.Tensor,
        report_attention_mask: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> dict:
        """
        Full forward pass for training.

        Sequence layout:
          [system_prompt | user_start] [image × 256] [user_end | asst_start] [report <|im_end|>]
           ← prefix_embeds →           ← image →     ← suffix_embeds →       ← report_embeds →
           labels: -100                 labels: -100  labels: -100            labels: real IDs
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # ---- 1. Encode image ----
        image_embeds = self.encode_image(pixel_values)  # [B, 256, 3584]

        # ---- 2. Get text embeddings ----
        embed_fn = self.language_model.get_input_embeddings()

        prefix_embeds = embed_fn(self.prefix_ids.expand(B, -1))   # [B, T1, 3584]
        suffix_embeds = embed_fn(self.suffix_ids.expand(B, -1))   # [B, T2, 3584]
        report_embeds = embed_fn(report_ids)                       # [B, T3, 3584]

        T1 = prefix_embeds.shape[1]
        T_img = image_embeds.shape[1]  # 256
        T2 = suffix_embeds.shape[1]
        T3 = report_ids.shape[1]

        # ---- 3. Concatenate all embeddings ----
        inputs_embeds = torch.cat(
            [prefix_embeds, image_embeds, suffix_embeds, report_embeds], dim=1
        ).to(dtype=self.llm_dtype)  # [B, T_total, 3584]

        # ---- 4. Build attention mask ----
        prompt_mask = torch.ones(B, T1 + T_img + T2, dtype=torch.long, device=device)
        attention_mask = torch.cat([prompt_mask, report_attention_mask], dim=1)

        # ---- 5. Build labels ----
        prompt_labels = torch.full(
            (B, T1 + T_img + T2), fill_value=-100, dtype=torch.long, device=device
        )
        report_labels = report_ids.clone()
        report_labels[report_attention_mask == 0] = -100
        labels = torch.cat([prompt_labels, report_labels], dim=1)

        # ---- 6. Forward through LLM ----
        if label_smoothing == 0.0:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = self._label_smoothed_loss(logits, labels, smoothing=label_smoothing)

        return {"loss": loss, "logits": logits}

    def _label_smoothed_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.1
    ) -> torch.Tensor:
        """Cross-entropy loss with label smoothing."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        mask = shift_labels != -100
        active_logits = shift_logits[mask]
        active_labels = shift_labels[mask]

        if active_labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = F.log_softmax(active_logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, active_labels, reduction="mean")
        smooth_loss = -log_probs.mean(dim=-1).mean()

        return (1.0 - smoothing) * nll_loss + smoothing * smooth_loss

    @torch.no_grad()
    def generate_report(
        self,
        pixel_values: torch.Tensor,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = cfg.GEN_MAX_NEW_TOKENS,
        num_beams: int = cfg.GEN_NUM_BEAMS,
        repetition_penalty: float = cfg.GEN_REPETITION_PENALTY,
        length_penalty: float = cfg.GEN_LENGTH_PENALTY,
    ) -> list:
        """Generate radiology reports for given images."""
        B = pixel_values.shape[0]
        device = pixel_values.device

        image_embeds = self.encode_image(pixel_values)

        embed_fn = self.language_model.get_input_embeddings()
        prefix_embeds = embed_fn(self.prefix_ids.expand(B, -1))
        suffix_embeds = embed_fn(self.suffix_ids.expand(B, -1))

        inputs_embeds = torch.cat(
            [prefix_embeds, image_embeds, suffix_embeds], dim=1
        ).to(dtype=self.llm_dtype)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )

        prompt_length = inputs_embeds.shape[1]

        # Use <|im_end|> as the EOS token for generation
        eos_token_id = self.im_end_token_id

        output_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        if output_ids.shape[1] > max_new_tokens:
            generated_ids = output_ids[:, prompt_length:]
        else:
            generated_ids = output_ids

        reports = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        cleaned = []
        for text in reports:
            # Clean up any ChatML artifacts
            for tag in ["<|im_start|>", "<|im_end|>", "assistant", "user", "system"]:
                text = text.replace(tag, "")
            cleaned.append(text.strip())

        return cleaned

    # ================================================================
    # Training stage control
    # ================================================================

    def freeze_for_stage1(self):
        """Stage 1: Train ONLY the projector."""
        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.projector.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Stage 1] Trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.4f}%)")

    def prepare_for_stage2(self):
        """Stage 2: Train projector + LoRA adapters on LLM."""
        for param in self.projector.parameters():
            param.requires_grad = True

        lora_config = LoraConfig(
            r=cfg.LORA_RANK,
            lora_alpha=cfg.LORA_ALPHA,
            target_modules=cfg.LORA_TARGET_MODULES,
            lora_dropout=cfg.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.print_trainable_parameters()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Stage 2] Total trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.4f}%)")

    def save_checkpoint(self, path: str, stage: int):
        """Save projector weights and (if stage 2) LoRA adapters."""
        import os
        os.makedirs(path, exist_ok=True)

        torch.save(
            self.projector.state_dict(),
            os.path.join(path, "projector.pth"),
        )

        if stage == 2:
            lora_path = os.path.join(path, "lora_adapters")
            self.language_model.save_pretrained(lora_path)

        import json
        with open(os.path.join(path, "model_info.json"), "w") as f:
            json.dump({
                "stage": stage,
                "raddino_model": cfg.RADDINO_MODEL_NAME,
                "llm_model": cfg.LLM_MODEL_NAME,
                "raddino_hidden_size": cfg.RADDINO_HIDDEN_SIZE,
                "llm_hidden_size": cfg.LLM_HIDDEN_SIZE,
                "num_image_tokens": cfg.NUM_IMAGE_TOKENS,
            }, f, indent=2)

        print(f"[Checkpoint] Saved stage {stage} checkpoint to {path}")

    def load_checkpoint(self, path: str, tokenizer: AutoTokenizer):
        """Load projector weights and LoRA adapters from a checkpoint."""
        import os

        projector_path = os.path.join(path, "projector.pth")
        if os.path.exists(projector_path):
            self.projector.load_state_dict(
                torch.load(projector_path, map_location="cpu", weights_only=True)
            )
            print(f"[Checkpoint] Loaded projector from {projector_path}")

        lora_path = os.path.join(path, "lora_adapters")
        if os.path.isdir(lora_path):
            from peft import PeftModel
            self.language_model = PeftModel.from_pretrained(
                self.language_model, lora_path
            )
            print(f"[Checkpoint] Loaded LoRA adapters from {lora_path}")
