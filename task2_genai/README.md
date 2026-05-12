# task2_genai/README.md

# Task 2 — Lightweight QLoRA Fine-Tuning Workflow

## Overview

This task implements a practical notebook-oriented QLoRA fine-tuning workflow using:
- Hugging Face Transformers
- PEFT
- TRL
- BitsAndBytes quantization
- and the Hugging Face Hub ecosystem.

The workflow demonstrates parameter-efficient instruction fine-tuning using LoRA adapters on limited hardware.

Rather than focusing on large-scale benchmark optimization, the primary goals were:
- reproducibility
- workflow clarity
- practical engineering
- resource-aware design
- and notebook-friendly experimentation.

The default model used is:

```text
Qwen/Qwen2.5-3B-Instruct
```

LoRA adapters are saved under:

```text
task2_genai/outputs/
```

---

# Workflow Structure

```text
Instruction Dataset Creation
            ↓
Baseline Model Evaluation
            ↓
4-bit Quantized Model Loading
            ↓
QLoRA Adapter Preparation
            ↓
LoRA Fine-Tuning
            ↓
Post-Training Evaluation
            ↓
Adapter Export / Hugging Face Upload
```

---

# Main Features

## QLoRA Fine-Tuning

The workflow uses:
- 4-bit quantization
- LoRA adapters
- and parameter-efficient fine-tuning

to reduce GPU memory usage and make training feasible on free-tier Colab hardware.

---

## Reproducible Configuration

Training settings are stored inside:
- `training_config.json`

This keeps experiments:
- reproducible
- inspectable
- and easier to modify.

---

## Baseline vs Fine-Tuned Evaluation

The workflow generates:
- baseline model responses
- fine-tuned model responses
- and structured before/after evaluation outputs.

This helps make workflow behavior easier to inspect qualitatively.

---

## Adapter-Based Training

The workflow fine-tunes lightweight LoRA adapters instead of modifying the full base model.

This dramatically reduces:
- storage requirements
- GPU memory usage
- and training cost.

---

## Hugging Face Integration

The notebook supports:
- Hugging Face login
- adapter publishing
- and model artifact export.

This demonstrates practical integration with modern LLM tooling ecosystems.

---

# Included Artifacts

## `training_config.json`

Reproducible training configuration including:
- learning rate
- LoRA configuration
- quantization settings
- and training hyperparameters.

---

## `data/instruction_dataset.jsonl`

Small curated instruction dataset used for lightweight instruction tuning.

The dataset is intentionally compact to keep the workflow practical for limited hardware environments.

---

## `evaluation/before_vs_after_results.json`

Structured comparison between:
- baseline model outputs
- and fine-tuned outputs.

---

## `model_card.md`

Contains:
- intended use
- workflow limitations
- training notes
- and deployment considerations.

---

# Running the Workflow

The workflow is designed primarily for Google Colab GPU environments.

Recommended runtime:
- T4 GPU
- Python 3.10+
- free-tier Colab environment.

Install dependencies:

```bash
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece
```

Run the notebook step-by-step:
- dataset preparation
- model loading
- baseline evaluation
- QLoRA training
- final evaluation
- adapter export.

---

# Engineering Notes

Task 2 was intentionally designed as a lightweight and practical engineering workflow rather than a large-scale research experiment.

One important lesson during implementation was how rapidly the Hugging Face ecosystem evolves.

A significant part of the workflow involved:
- dependency compatibility handling
- Colab environment debugging
- API version stabilization
- and practical GPU resource management.

The trainer configuration was intentionally simplified to improve reproducibility and compatibility across free-tier notebook environments.

---

# Limitations

- Small datasets limit final model capability.
- Free-tier GPU environments restrict training scale.
- Outputs may vary slightly between runtime environments.
- Hugging Face ecosystem version changes may require minor compatibility adjustments over time.
- This workflow is educational and engineering-focused rather than production-scale LLM training.

---

# Main Takeaways

This task reinforced the importance of:
- reproducible experimentation
- parameter-efficient fine-tuning
- operational debugging
- dependency management
- and realistic resource-aware AI engineering.