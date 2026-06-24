# GPT2-ChineseDevBench

This repository contains a Chinese GPT-2 curriculum training experiment based on `uer/gpt2-chinese-cluecorpussmall`.

The model is trained in two stages:

* Stage 1: 0-3 years corpus
* Stage 2: 3-6 years corpus, continued from Stage 1

The final trained model is saved under:

```text
model/stage2_final_model/
```

## Directory Structure

```text
GPT2-ChineseDevBench/
├── README.md
├── requirements.txt
├── baseline/
├── dataset/
│   ├── raw_datasets/
│   │   ├── transcriptions_0-3岁/
│   │   └── transcriptions_4-6岁/
│   └── processed_datasets/
│       ├── corpus_stage1_0-3.txt
│       └── corpus_stage2_3-6.txt
├── model/
│   ├── stage1_final_model/
│   └── stage2_final_model/
└── scripts/
    └── train_baseline_four_datasets.py
```

## Environment

Python 3.10 is recommended.

```bash
conda create -n gpt2-chinese-devbench python=3.10 -y
conda activate gpt2-chinese-devbench
pip install -r requirements.txt
```

## Use the Trained Model

The final trained model is located at:

```text
model/stage2_final_model/
```

Example inference:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "model/stage2_final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

prompt = "小朋友说"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
PY
```

## Training Data

The training script uses four data sources:

```text
dataset/raw_datasets/transcriptions_0-3岁/
dataset/raw_datasets/transcriptions_4-6岁/
dataset/processed_datasets/corpus_stage1_0-3.txt
dataset/processed_datasets/corpus_stage2_3-6.txt
```

Stage 1 uses:

```text
dataset/raw_datasets/transcriptions_0-3岁/
dataset/processed_datasets/corpus_stage1_0-3.txt
```

Stage 2 uses:

```text
dataset/raw_datasets/transcriptions_4-6岁/
dataset/processed_datasets/corpus_stage2_3-6.txt
```

## Re-train

Run:

```bash
python scripts/train_baseline_four_datasets.py
```

The script trains Stage 1 first, then continues Stage 2 from the Stage 1 final model.

New outputs will be saved to:

```text
output_baseline_four/
```

The newly trained final model will be:

```text
output_baseline_four/stage2_3-6/final_model/
```

## Resume Training

If training is interrupted, run the same command again:

```bash
python scripts/train_baseline_four_datasets.py
```

The script will resume from the latest checkpoint.

## Main Hyperparameters

| Item                  | Value                            |
| --------------------- | -------------------------------- |
| Model type            | Decoder-only GPT-2               |
| Base model            | uer/gpt2-chinese-cluecorpussmall |
| Optimizer             | AdamW                            |
| Tokenizer             | WordPiece                        |
| Vocab size            | 21128                            |
| Layers                | 12                               |
| Attention heads       | 12                               |
| Hidden size           | 768                              |
| Max sequence length   | 1024                             |
| Parameters            | about 102M                       |
| Epochs                | 10                               |
| Batch size            | 8                                |
| Stage 1 learning rate | 1e-4                             |
| Stage 2 learning rate | 5e-5                             |
| Evaluation split      | None                             |
| Save strategy         | epoch                            |

## Notes

This experiment does not use a 9:1 train/eval split.

The packaged model under `model/stage2_final_model/` can be used directly without re-training.
