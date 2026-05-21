"""
Inference Script for Qwen-RADDino.

Usage:
  python inference.py --checkpoint /path/to/stage2_best --output results.json
"""

import os
import sys
import json
import argparse
import time

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

import config as cfg
from model import QwenRaddino
from dataset import IUXrayDataset, collate_fn
from raddino_encoder import RadDinoEncoder
from metrics import compute_all_metrics, aggregate_metrics
from functools import partial
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Qwen-RADDino inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.GEN_MAX_NEW_TOKENS)
    parser.add_argument("--num_beams", type=int, default=cfg.GEN_NUM_BEAMS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Qwen2.5 tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Build & load model ----
    print(f"\nBuilding model...")
    model = QwenRaddino(tokenizer=tokenizer)
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_checkpoint(args.checkpoint, tokenizer)
    model = model.to(device)
    model.eval()

    # ---- Test dataset ----
    transform = RadDinoEncoder.get_image_transform()
    test_dataset = IUXrayDataset(
        json_path=cfg.TEST_JSON,
        image_dir=cfg.IMAGE_DIR,
        tokenizer=tokenizer,
        transform=transform,
        max_report_len=cfg.MAX_REPORT_LENGTH,
    )
    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=_collate, pin_memory=True,
    )
    print(f"Test set: {len(test_dataset)} samples")

    # ---- Run inference ----
    all_results = []
    all_metrics_list = []
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating reports"):
            pixel_values = batch["pixel_values"].to(device)
            ground_truths = batch["ground_truth"]
            sample_ids = batch["sample_id"]
            image_paths = batch["image_path"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                generated_reports = model.generate_report(
                    pixel_values=pixel_values,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )

            for i in range(len(generated_reports)):
                sample_metrics = compute_all_metrics(ground_truths[i], generated_reports[i])
                all_metrics_list.append(sample_metrics)
                all_results.append({
                    "id": sample_ids[i],
                    "image_path": image_paths[i],
                    "ground_truth": ground_truths[i],
                    "generated_report": generated_reports[i],
                    "metrics": sample_metrics,
                })

    elapsed = time.time() - start_time
    agg_metrics = aggregate_metrics(all_metrics_list)

    output = {
        "model_info": {
            "vision_encoder": cfg.RADDINO_MODEL_NAME,
            "language_model": cfg.LLM_MODEL_NAME,
            "checkpoint": args.checkpoint,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
        },
        "aggregate_metrics": agg_metrics,
        "num_samples": len(all_results),
        "inference_time_seconds": round(elapsed, 2),
        "results": all_results,
    }

    output_path = args.output if args.output.endswith(".json") else args.output + ".json"
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE -- {len(all_results)} samples in {elapsed:.1f}s")
    print(f"{'='*60}")
    for k, v in agg_metrics.items():
        print(f"  {k:>10}: {v:.4f}")
    print(f"\nResults saved to: {output_path}")

    for i, r in enumerate(all_results[:3]):
        print(f"\n--- Sample {i+1} (ID: {r['id']}) ---")
        print(f"  GT:  {r['ground_truth'][:200]}")
        print(f"  GEN: {r['generated_report'][:200]}")
        print(f"  BLEU-4: {r['metrics']['bleu_4']:.4f}  ROUGE-L: {r['metrics']['rouge_l']:.4f}")


if __name__ == "__main__":
    main()
