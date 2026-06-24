#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import hashlib
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


try:
    import zhconv
except Exception:
    zhconv = None


EOS_TOKEN = "<|endoftext|>"


# ============================================================
# 只使用你指定的四个训练源
# 1. 0-3岁原始转录目录
# 2. 4-6岁原始转录目录
# 3. corpus_stage1_0-3.txt
# 4. corpus_stage2_3-6.txt
# ============================================================

STAGE_DATA = {
    "stage1_0-3": {
        "raw_dir_candidates": [
            "dataset/raw_datasets/transcriptions_0-3岁",
            "dataset/data0/datasets/videos/视频爬取/downloads/transcriptions/0-3岁",
        ],
        "extra_file_candidates": [
            "dataset/processed_datasets/corpus_stage1_0-3.txt",
        ],
    },
    "stage2_3-6": {
        "raw_dir_candidates": [
            "dataset/raw_datasets/transcriptions_4-6岁",
            "dataset/data0/datasets/videos/视频爬取/downloads/transcriptions/4-6岁",
        ],
        "extra_file_candidates": [
            "dataset/processed_datasets/corpus_stage2_3-6.txt",
        ],
    },
}


BASE_MODEL_CANDIDATES = [
    "models--uer--gpt2-chinese-cluecorpussmall",
    "dataset/models--uer--gpt2-chinese-cluecorpussmall",
    "dataset/huggingface/models--uer--gpt2-chinese-cluecorpussmall",
    "/home/czs/.cache/huggingface/hub/models--uer--gpt2-chinese-cluecorpussmall",
    "uer/gpt2-chinese-cluecorpussmall",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: str) -> Path:
    return project_root() / path


def file_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if zhconv is not None:
        text = zhconv.convert(text, "zh-cn")

    return text.strip()


def extract_from_txt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")

        separator = "--------------------------------------------------"
        if separator in text:
            text = text.split(separator)[-1]

        return clean_text(text)

    except Exception:
        return ""


def extract_from_json(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return ""

        main_audio_text = ""
        parts = []

        for item in data:
            if not isinstance(item, dict):
                continue

            raw_text = item.get("text", "")
            page_info = str(item.get("page_info", ""))

            text = clean_text(raw_text)
            if not text:
                continue

            if "主音频" in page_info:
                main_audio_text = text
            else:
                parts.append(text)

        if main_audio_text and len(main_audio_text) > 5:
            return main_audio_text

        return clean_text("".join(parts))

    except Exception:
        return ""


def normalize_training_text(text: str) -> str:
    text = clean_text(text)
    text = text.replace(EOS_TOKEN, "").strip()
    return text


def first_existing_dir(candidates):
    for c in candidates:
        p = rel(c)
        if p.is_dir():
            return p
    return None


def existing_files(candidates):
    out = []
    for c in candidates:
        p = rel(c)
        if p.is_file() and p.stat().st_size > 0:
            out.append(p)
    return out


def resolve_base_model():
    for c in BASE_MODEL_CANDIDATES:
        p = Path(c)

        if not p.is_absolute():
            p = rel(c)

        if p.is_dir():
            if (p / "config.json").exists:
                if (p / "config.json").is_file():
                    return str(p)

            snapshots = sorted(glob.glob(str(p / "snapshots" / "*")))
            for s in snapshots:
                if Path(s, "config.json").is_file():
                    return s

    return "uer/gpt2-chinese-cluecorpussmall"


def collect_raw_files(raw_dir: Path):
    files = []
    for root, _, names in os.walk(raw_dir):
        for name in names:
            if name.endswith(".txt") or name.endswith(".json"):
                p = Path(root) / name
                if p.is_file() and p.stat().st_size > 0:
                    files.append(p)
    return sorted(files)


def collect_stage_texts(stage_name: str):
    cfg = STAGE_DATA[stage_name]

    texts = []
    seen = set()
    source_info = {
        "stage": stage_name,
        "raw_dir": None,
        "raw_file_count": 0,
        "extra_files": [],
        "unique_text_count": 0,
    }

    raw_dir = first_existing_dir(cfg["raw_dir_candidates"])
    if raw_dir is None:
        raise FileNotFoundError(f"{stage_name} 找不到原始目录: {cfg['raw_dir_candidates']}")

    source_info["raw_dir"] = str(raw_dir.relative_to(project_root()))
    raw_files = collect_raw_files(raw_dir)
    source_info["raw_file_count"] = len(raw_files)

    print(f"\n[{stage_name}] raw_dir = {source_info['raw_dir']}")
    print(f"[{stage_name}] raw_files = {len(raw_files)}")

    for p in tqdm(raw_files, desc=f"读取原始目录 {stage_name}"):
        if p.suffix == ".json":
            text = extract_from_json(p)
        else:
            text = extract_from_txt(p)

        text = normalize_training_text(text)
        if len(text) <= 5:
            continue

        h = file_md5(text)
        if h not in seen:
            seen.add(h)
            texts.append(text)

    extra_files = existing_files(cfg["extra_file_candidates"])
    if not extra_files:
        raise FileNotFoundError(f"{stage_name} 找不到额外 corpus 文件: {cfg['extra_file_candidates']}")

    for fp in extra_files:
        source_info["extra_files"].append(str(fp.relative_to(project_root())))
        print(f"[{stage_name}] extra_file = {fp.relative_to(project_root())}")

        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc=f"读取额外语料 {fp.name}"):
                text = normalize_training_text(line)
                if len(text) <= 5:
                    continue

                h = file_md5(text)
                if h not in seen:
                    seen.add(h)
                    texts.append(text)

    source_info["unique_text_count"] = len(texts)

    print(f"[{stage_name}] unique_texts = {len(texts)}")

    if len(texts) == 0:
        raise RuntimeError(f"{stage_name} 没有读取到任何有效训练文本")

    return texts, source_info


class CausalLMDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size: int):
        self.block_size = block_size
        all_ids = []

        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise RuntimeError("tokenizer.eos_token_id is None")

        for text in tqdm(texts, desc="分词 tokenizing"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                all_ids.extend(ids)
                all_ids.append(eos_id)

        total_tokens = len(all_ids)

        usable_tokens = (total_tokens // block_size) * block_size
        all_ids = all_ids[:usable_tokens]

        self.examples = []
        for i in tqdm(range(0, usable_tokens, block_size), desc="切 block"):
            block = torch.tensor(all_ids[i:i + block_size], dtype=torch.long)
            self.examples.append(block)

        print(f"[Dataset] total_tokens = {total_tokens}")
        print(f"[Dataset] usable_tokens = {usable_tokens}")
        print(f"[Dataset] blocks = {len(self.examples)}")
        print(f"[Dataset] block_size = {block_size}")

        if len(self.examples) == 0:
            raise RuntimeError("训练 block 数量为 0，检查数据或 block_size")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x.clone(),
        }


def prepare_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # 旧实验要求 vocab_size=21128，所以这里不新增 token，避免 resize 后变成 21129。
    # 优先用已有 eos_token；没有就用 sep_token；再没有就用 pad/unk。
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is not None:
            tokenizer.eos_token = tokenizer.sep_token
        elif tokenizer.pad_token is not None:
            tokenizer.eos_token = tokenizer.pad_token
        elif tokenizer.unk_token is not None:
            tokenizer.eos_token = tokenizer.unk_token
        else:
            raise RuntimeError("tokenizer 没有 eos/sep/pad/unk token，无法安全设置 EOS。")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, False


def train_one_stage(stage_name: str, model_path: str, tokenizer_path: str, args):
    output_dir = rel(args.output_dir) / stage_name
    final_model_dir = output_dir / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    if final_model_dir.is_dir() and not args.force:
        print(f"\n[SKIP] {stage_name} 已有 final_model: {final_model_dir.relative_to(project_root())}")
        return str(final_model_dir)

    texts, source_info = collect_stage_texts(stage_name)

    tokenizer, _ = prepare_tokenizer(tokenizer_path)

    print(f"\n[{stage_name}] 加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = CausalLMDataset(
        texts=texts,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )

    manifest_path = output_dir / "data_manifest.json"
    manifest = {
        "stage": stage_name,
        "model_input": model_path,
        "tokenizer_input": tokenizer_path,
        "output_dir": str(output_dir.relative_to(project_root())),
        "final_model_dir": str(final_model_dir.relative_to(project_root())),
        "block_size": args.block_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "save_strategy": "epoch",
        "save_total_limit": None,
        "resume": True,
        "source_info": source_info,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=(1e-4 if stage_name.startswith("stage1") else 5e-5),
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=None,

        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=False,

        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    last_ckpt = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None

    if last_ckpt:
        print(f"\n[RESUME] {stage_name} 从 checkpoint 继续: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print(f"\n[START] {stage_name} 从当前模型开始训练")
        trainer.train()

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\n[SAVED] {stage_name}: {final_model_dir.relative_to(project_root())}")

    return str(final_model_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="output_baseline_four")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    print("========== GPT2 Chinese Baseline Curriculum Training ==========")
    print(f"project_root = {project_root()}")
    print(f"output_dir   = {args.output_dir}")
    print(f"epochs       = {args.epochs}")
    print(f"block_size   = {args.block_size}")
    print(f"batch_size   = {args.batch_size}")
    print(f"grad_accum   = {args.grad_accum}")
    print("lr           = stage1: 1e-4, stage2: 5e-5")

    base_model = resolve_base_model()
    print(f"\n[BASELINE] {base_model}")

    stage1_model = train_one_stage(
        stage_name="stage1_0-3",
        model_path=base_model,
        tokenizer_path=base_model,
        args=args,
    )

    stage2_model = train_one_stage(
        stage_name="stage2_3-6",
        model_path=stage1_model,
        tokenizer_path=stage1_model,
        args=args,
    )

    print("\n========== DONE ==========")
    print(f"Final model: {Path(stage2_model).relative_to(project_root())}")


if __name__ == "__main__":
    main()
