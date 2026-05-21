"""
Two-Stage Training Script for Qwen-RADDino.

Stage 1: Trains ONLY the projector MLP (768->3584->3584)
Stage 2: Trains projector + LoRA adapters on Qwen2.5-7B

Usage:
  python train.py --stage 1
  python train.py --stage 2 --stage1_ckpt /path/to/stage1/checkpoint
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime

import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

import config as cfg
from model import QwenRaddino
from dataset import create_dataloaders
from raddino_encoder import RadDinoEncoder


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(model, val_loader, device, label_smoothing=0.0):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            report_ids = batch["report_ids"].to(device)
            report_mask = batch["report_attention_mask"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=pixel_values,
                    report_ids=report_ids,
                    report_attention_mask=report_mask,
                    label_smoothing=label_smoothing,
                )
            total_loss += outputs["loss"].item()
            num_batches += 1
    model.train()
    return total_loss / max(num_batches, 1)


def generate_samples(model, val_loader, tokenizer, device, num_samples=3):
    model.eval()
    batch = next(iter(val_loader))
    pixel_values = batch["pixel_values"][:num_samples].to(device)
    ground_truths = batch["ground_truth"][:num_samples]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate_report(pixel_values, tokenizer)
    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS (diversity check)")
    print("=" * 70)
    for i in range(min(num_samples, len(generated))):
        print(f"\n--- Sample {i+1} ---")
        print(f"  GT:  {ground_truths[i][:200]}...")
        print(f"  GEN: {generated[i][:200]}...")
    print("=" * 70)
    unique_outputs = len(set(generated))
    if unique_outputs == 1 and num_samples > 1:
        print("WARNING: All outputs identical! Possible collapse.")
    else:
        print(f"Diversity: {unique_outputs}/{num_samples} unique")
    model.train()


def train_stage(model, train_loader, val_loader, tokenizer, device, stage, output_dir):
    if stage == 1:
        lr = cfg.STAGE1_LR
        epochs = cfg.STAGE1_EPOCHS
        warmup_ratio = cfg.STAGE1_WARMUP_RATIO
        weight_decay = cfg.STAGE1_WEIGHT_DECAY
        label_smoothing = 0.0
        max_grad_norm = None
        patience = None
        grad_accum_steps = 1
    else:
        lr = cfg.STAGE2_LR
        epochs = cfg.STAGE2_EPOCHS
        warmup_ratio = cfg.STAGE2_WARMUP_RATIO
        weight_decay = cfg.STAGE2_WEIGHT_DECAY
        label_smoothing = cfg.STAGE2_LABEL_SMOOTHING
        max_grad_norm = cfg.STAGE2_MAX_GRAD_NORM
        patience = cfg.STAGE2_PATIENCE
        grad_accum_steps = cfg.STAGE2_GRAD_ACCUM_STEPS

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"STAGE {stage} TRAINING -- Qwen-RADDino")
    print(f"{'='*70}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {train_loader.batch_size}x{grad_accum_steps}")
    print(f"  Steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"  Label smoothing: {label_smoothing}, Grad clip: {max_grad_norm}")
    print(f"  Params: {trainable:,} trainable / {total:,} total")
    print(f"  Projector: {cfg.RADDINO_HIDDEN_SIZE} -> {cfg.LLM_HIDDEN_SIZE}")
    print(f"{'='*70}\n")

    model.train()
    model.vision_encoder.model.eval()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Stage {stage} | Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            report_ids = batch["report_ids"].to(device)
            report_mask = batch["report_attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=pixel_values,
                    report_ids=report_ids,
                    report_attention_mask=report_mask,
                    label_smoothing=label_smoothing,
                )
                loss = outputs["loss"] / grad_accum_steps

            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += outputs["loss"].item()
            epoch_steps += 1
            pbar.set_postfix(loss=f"{outputs['loss'].item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train = epoch_loss / max(epoch_steps, 1)
        val_loss = validate(model, val_loader, device, label_smoothing)

        print(f"\nEpoch {epoch}/{epochs}: train={avg_train:.4f}, val={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        generate_samples(model, val_loader, tokenizer, device, num_samples=3)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_checkpoint(os.path.join(output_dir, f"stage{stage}_best"), stage)
            print(f"  New best val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience or 'inf'}")

        model.save_checkpoint(os.path.join(output_dir, f"stage{stage}_latest"), stage)

        if patience and patience_counter >= patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break

    print(f"\nSTAGE {stage} COMPLETE -- Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Qwen-RADDino")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=cfg.OUTPUT_DIR)
    args = parser.parse_args()

    if args.stage == 2 and not args.stage1_ckpt:
        print("ERROR: --stage1_ckpt required for Stage 2")
        sys.exit(1)

    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Qwen2.5 tokenizer ----
    print(f"\nLoading tokenizer: {cfg.LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL_NAME)

    # Qwen2.5 tokenizer: set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    transform = RadDinoEncoder.get_image_transform()

    batch_size = cfg.STAGE1_BATCH_SIZE if args.stage == 1 else cfg.STAGE2_BATCH_SIZE
    print(f"\nLoading datasets...")
    train_loader, val_loader, _ = create_dataloaders(
        tokenizer=tokenizer,
        transform=transform,
        batch_size_train=batch_size,
        batch_size_eval=batch_size,
        num_workers=cfg.NUM_WORKERS,
    )
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    print(f"\nBuilding model...")
    model = QwenRaddino(tokenizer=tokenizer)

    if args.stage == 1:
        model.freeze_for_stage1()
    elif args.stage == 2:
        print(f"\nLoading Stage 1 checkpoint: {args.stage1_ckpt}")
        model.load_checkpoint(args.stage1_ckpt, tokenizer)
        model.prepare_for_stage2()

    model = model.to(device)

    train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        stage=args.stage,
        output_dir=args.output_dir,
    )

    info = {
        "stage": args.stage,
        "completed_at": datetime.now().isoformat(),
        "output_dir": args.output_dir,
        "stage1_ckpt": args.stage1_ckpt,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
    }
    with open(os.path.join(args.output_dir, f"stage{args.stage}_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
