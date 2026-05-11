# Task 2: Lightweight QLoRA Instruction Fine-Tuning

This task implements a notebook-oriented instruction fine-tuning workflow using:

- `transformers`
- `peft`
- `trl`
- `bitsandbytes`
- `datasets`

The default model is `Qwen/Qwen2.5-3B-Instruct`, with LoRA adapters saved under
`task2_genai/outputs/`.

## Why This Design

The workflow is intentionally lightweight and explicit. It demonstrates practical
GenAI engineering without distributed training, benchmark chasing, or unnecessary
framework layers.

## Main Artifacts

- `notebook.ipynb`: step-by-step training and evaluation workflow
- `src/finetuning_workflow.py`: reusable helper functions
- `training_config.json`: reproducible hyperparameters
- `data/instruction_dataset.jsonl`: small instruction/response dataset
- `data/sample_prompts.json`: fixed evaluation prompts
- `evaluation/before_vs_after_results.json`: structured evaluation output
- `model_card.md`: intended use, limitations, and training notes

## Recommended Runtime

Use Google Colab with a free GPU runtime. The code uses 4-bit quantization and
LoRA adapters to reduce memory usage.

## Important Note

This is an educational fine-tuning workflow. The small dataset is designed to
show reproducible engineering practices, not to produce a broadly capable model.
