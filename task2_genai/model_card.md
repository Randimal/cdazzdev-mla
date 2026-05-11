# Model Card: Task 2 QLoRA Adapter

## Base Model

Default: `Qwen/Qwen2.5-3B-Instruct`

Alternative: `meta-llama/Llama-3.2-3B-Instruct` if access and hardware allow.

## Fine-Tuning Method

This workflow uses QLoRA:

- the base model is loaded in 4-bit precision with bitsandbytes
- the base weights remain frozen
- small LoRA adapter matrices are trained with PEFT
- adapters are saved separately from the base model

## Intended Use

This adapter is intended for educational demonstrations of:

- instruction fine-tuning
- parameter-efficient tuning
- reproducible GenAI workflows
- before/after qualitative evaluation

## Dataset

The dataset is a small instruction/response set focused on practical AI
engineering concepts such as QLoRA, structured outputs, PEFT, graceful
degradation, and reproducibility.

The dataset is intentionally small. It is meant to demonstrate adaptation and
workflow quality, not broad domain knowledge acquisition.

## Training Configuration

See `training_config.json` for:

- base model
- LoRA rank and alpha
- learning rate
- batch size
- gradient accumulation
- sequence length
- quantization settings
- adapter output path

## Limitations

- The dataset is very small.
- The model may overfit to the style of the examples.
- The adapter should not be treated as a production-ready assistant.
- Free-tier GPUs may run out of memory depending on runtime availability.
- Outputs can still be incomplete, incorrect, or overconfident.

## Evaluation

Evaluation is qualitative and compares fixed prompts before and after
fine-tuning. Results are saved to:

`task2_genai/evaluation/before_vs_after_results.json`

## Production Considerations

A production workflow would add:

- larger curated datasets
- held-out validation data
- automated evaluation
- toxicity/safety checks
- model registry integration
- reproducible environment pinning
- adapter versioning
- monitoring after deployment
