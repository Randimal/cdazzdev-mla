import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK2_DIR = PROJECT_ROOT / "task2_genai"
DATA_DIR = TASK2_DIR / "data"
EVALUATION_DIR = TASK2_DIR / "evaluation"
OUTPUTS_DIR = TASK2_DIR / "outputs"
CONFIG_PATH = TASK2_DIR / "training_config.json"

JSONDict = Dict[str, Any]


DEFAULT_TRAINING_CONFIG: JSONDict = {
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "adapter_output_dir": "task2_genai/outputs/qwen2_5_3b_task2_lora",
    "dataset_path": "task2_genai/data/instruction_dataset.jsonl",
    "evaluation_path": "task2_genai/evaluation/before_vs_after_results.json",
    "sample_prompts_path": "task2_genai/data/sample_prompts.json",
    "max_seq_length": 768,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "logging_steps": 5,
    "save_strategy": "epoch",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "seed": 42,
    "max_new_tokens": 220,
}


SEED_INSTRUCTION_EXAMPLES: List[JSONDict] = [
    {
        "instruction": "Explain QLoRA to a junior machine learning engineer.",
        "response": (
            "QLoRA fine-tunes a quantized language model using small LoRA "
            "adapter weights. The base model is loaded in 4-bit precision to "
            "reduce GPU memory, while only the adapter matrices are trained. "
            "This makes instruction tuning practical on limited hardware."
        ),
    },
    {
        "instruction": "Summarize why structured outputs improve AI workflows.",
        "response": (
            "Structured outputs make model responses easier to validate, log, "
            "test, and pass between components. Instead of parsing free-form "
            "text, downstream code can rely on predictable fields and handle "
            "missing or invalid values safely."
        ),
    },
    {
        "instruction": "Give three production risks in an LLM application.",
        "response": (
            "Three production risks are malformed model outputs, hidden prompt "
            "or data drift, and external API failures. Good systems mitigate "
            "these with validation, observability, fallbacks, and monitoring."
        ),
    },
    {
        "instruction": "Explain why small high-quality datasets can work for instruction tuning.",
        "response": (
            "Small high-quality datasets can work when the goal is style, "
            "format, or domain adaptation rather than teaching broad new "
            "knowledge. Clear examples help the model learn the expected answer "
            "shape without requiring thousands of noisy samples."
        ),
    },
    {
        "instruction": "Compare LoRA fine-tuning with full fine-tuning.",
        "response": (
            "LoRA fine-tuning updates small adapter matrices while keeping the "
            "base model frozen, so it is cheaper and easier to store. Full "
            "fine-tuning updates all model weights and can be more flexible, "
            "but it requires much more memory, compute, and operational care."
        ),
    },
    {
        "instruction": "Write a concise model-card limitation for a fine-tuned assistant.",
        "response": (
            "This model is intended for educational demonstrations and may "
            "produce incomplete or incorrect answers. It was trained on a small "
            "dataset, so outputs should be reviewed before production use."
        ),
    },
    {
        "instruction": "Explain graceful degradation in an AI system.",
        "response": (
            "Graceful degradation means the system returns a safe, visible "
            "fallback when a component fails. For example, if an LLM API is "
            "unavailable, the workflow can return a structured error result "
            "instead of crashing or fabricating an answer."
        ),
    },
    {
        "instruction": "Describe a reproducible fine-tuning experiment.",
        "response": (
            "A reproducible fine-tuning experiment records the base model, "
            "dataset version, random seed, hyperparameters, adapter output "
            "path, evaluation prompts, and before/after generations."
        ),
    },
    {
        "instruction": "What is PEFT and why is it useful?",
        "response": (
            "PEFT means parameter-efficient fine-tuning. It trains a small "
            "number of additional parameters instead of updating the full model, "
            "which reduces GPU memory, storage cost, and training time."
        ),
    },
    {
        "instruction": "Explain quantization in practical terms.",
        "response": (
            "Quantization stores model weights with fewer bits, such as 4-bit "
            "instead of 16-bit. This reduces memory usage and makes larger "
            "models fit on smaller GPUs, with some tradeoff in numerical "
            "precision."
        ),
    },
    {
        "instruction": "Give advice for evaluating a small instruction-tuned model.",
        "response": (
            "Compare baseline and fine-tuned outputs on fixed prompts. Look for "
            "format adherence, clarity, factual caution, and whether the model "
            "matches the desired response style. Avoid claiming broad benchmark "
            "improvement from a tiny dataset."
        ),
    },
    {
        "instruction": "Explain why adapters are saved separately from the base model.",
        "response": (
            "Adapters are saved separately because they are small and contain "
            "only the fine-tuned changes. At inference time, the adapter is "
            "loaded on top of the original base model, which avoids duplicating "
            "billions of base-model parameters."
        ),
    },
]


SAMPLE_EVAL_PROMPTS: List[str] = [
    "Explain QLoRA in simple engineering terms.",
    "Why are structured outputs useful in production LLM systems?",
    "List common mistakes in notebook-based fine-tuning.",
    "Compare LoRA adapters with full fine-tuning.",
]


@dataclass
class GenerationRecord:
    """Small structured record used for before/after qualitative evaluation."""

    prompt: str
    baseline_response: Optional[str] = None
    fine_tuned_response: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> JSONDict:
        return {
            "prompt": self.prompt,
            "baseline_response": self.baseline_response,
            "fine_tuned_response": self.fine_tuned_response,
            "notes": self.notes,
        }


def ensure_task2_directories() -> None:
    """Create expected Task 2 artifact directories."""

    for directory in (DATA_DIR, EVALUATION_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> Path:
    """Save JSON artifacts with consistent formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
    return path


def resolve_project_path(path_like: str | Path) -> Path:
    """
    Resolve config paths relative to the repository root.

    Colab notebooks are often run from different working directories. Resolving
    paths here keeps artifacts in task2_genai/data, evaluation, and outputs
    regardless of where the notebook kernel starts.
    """

    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_config_paths(config: Mapping[str, Any]) -> JSONDict:
    """Return a config copy with filesystem paths expanded to absolute strings."""

    resolved = dict(config)
    for key in (
        "adapter_output_dir",
        "dataset_path",
        "evaluation_path",
        "sample_prompts_path",
    ):
        if key in resolved:
            resolved[key] = str(resolve_project_path(resolved[key]))
    return resolved


def load_json(path: Path) -> Any:
    """Load JSON artifacts."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def export_default_training_config(
    path: Path = CONFIG_PATH,
    overrides: Optional[Mapping[str, Any]] = None,
) -> JSONDict:
    """
    Write a reproducible training configuration.

    Keeping hyperparameters in JSON makes the notebook less fragile: training
    cells read one config object instead of hiding values across many cells.
    """

    config = dict(DEFAULT_TRAINING_CONFIG)
    if overrides:
        config.update(dict(overrides))
    save_json(path, config)
    return config


def load_training_config(path: Path = CONFIG_PATH) -> JSONDict:
    """Load the training config, creating the default file if needed."""

    if not path.exists():
        return export_default_training_config(path)
    return load_json(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write instruction examples as JSONL for easy dataset loading."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return path


def create_instruction_dataset(
    path: Path = DATA_DIR / "instruction_dataset.jsonl",
    examples: Optional[List[JSONDict]] = None,
) -> List[JSONDict]:
    """
    Create a small instruction/response dataset.

    The examples are intentionally compact and high quality. For this
    assessment, the goal is to demonstrate instruction-following adaptation and
    reproducible engineering, not to chase benchmark scores with a large noisy
    dataset.
    """

    dataset = examples or SEED_INSTRUCTION_EXAMPLES
    write_jsonl(path, dataset)
    return dataset


def save_sample_prompts(
    path: Path = DATA_DIR / "sample_prompts.json",
    prompts: Optional[List[str]] = None,
) -> List[str]:
    """Save fixed evaluation prompts so before/after comparisons are fair."""

    sample_prompts = prompts or SAMPLE_EVAL_PROMPTS
    save_json(path, sample_prompts)
    return sample_prompts


def build_chat_messages(instruction: str, response: Optional[str] = None) -> List[JSONDict]:
    """
    Build a simple instruction-tuning chat example.

    Using explicit user/assistant roles matches modern instruct-model training
    better than concatenating arbitrary strings by hand.
    """

    messages = [{"role": "user", "content": instruction}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def format_example_with_tokenizer(example: Mapping[str, str], tokenizer) -> str:
    """
    Convert an instruction/response row to model-ready text.

    tokenizer.apply_chat_template uses the base model's expected prompt format.
    This is safer than inventing a prompt format that may not match Qwen or
    Llama chat conventions.
    """

    messages = build_chat_messages(
        instruction=example["instruction"],
        response=example["response"],
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_instruction_dataset_for_training(dataset_path: str, tokenizer):
    """
    Load JSONL data with datasets and add a formatted text column.

    The import is lazy so the helper module remains importable even on machines
    where the fine-tuning stack is not installed yet.
    """

    from datasets import load_dataset

    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    def add_text_column(example: Mapping[str, str]) -> JSONDict:
        return {"text": format_example_with_tokenizer(example, tokenizer)}

    return raw_dataset.map(add_text_column)


def get_torch_dtype(dtype_name: str):
    """Map config strings to torch dtypes with a practical fallback."""

    import torch

    if dtype_name == "bfloat16" and torch.cuda.is_available():
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float16


def create_4bit_quantization_config(config: Mapping[str, Any]):
    """
    Build BitsAndBytes 4-bit quantization config.

    QLoRA uses quantization to fit a larger base model into limited GPU memory.
    The base weights are loaded in 4-bit precision, while LoRA adapter weights
    are trained in higher precision.
    """

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=bool(config.get("load_in_4bit", True)),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=get_torch_dtype(
            config.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
        bnb_4bit_use_double_quant=bool(
            config.get("bnb_4bit_use_double_quant", True)
        ),
    )


def load_base_model_and_tokenizer(config: Mapping[str, Any]):
    """
    Load the instruct model and tokenizer with 4-bit quantization.

    This function should be run on a GPU runtime. On CPU-only machines it may
    fail because bitsandbytes 4-bit kernels require compatible GPU support.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=create_4bit_quantization_config(config),
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def prepare_model_for_lora(model, config: Mapping[str, Any]):
    """
    Attach LoRA adapters to the quantized base model.

    PEFT keeps the base model frozen and trains small adapter matrices. This
    reduces memory, storage, and training cost while still allowing useful
    instruction-style adaptation.
    """

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=list(config.get("target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 220,
) -> str:
    """Generate one response for qualitative evaluation."""

    import torch

    messages = build_chat_messages(prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_model_outputs(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int = 220,
) -> List[GenerationRecord]:
    """Generate baseline or fine-tuned outputs for fixed prompts."""

    records = []
    for prompt in prompts:
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        records.append(
            GenerationRecord(
                prompt=prompt,
                baseline_response=response,
            )
        )
    return records


def train_qlora_adapters(model, tokenizer, train_dataset, config: Mapping[str, Any]):
    """
    Run lightweight QLoRA fine-tuning with TRL's SFTTrainer.

    The settings are intentionally conservative for free-tier GPUs: small batch
    size, gradient accumulation, one epoch, short sequence length, and LoRA
    adapters only. This demonstrates the workflow without pretending to be a
    large-scale training run.
    """

    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer

    output_dir = config["adapter_output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(
            config.get("per_device_train_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            config.get("gradient_accumulation_steps", 4)
        ),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        logging_steps=int(config.get("logging_steps", 5)),
        save_strategy=config.get("save_strategy", "epoch"),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=int(config.get("seed", 42)),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "dataset_text_field": "text",
        "max_seq_length": int(config.get("max_seq_length", 768)),
        "packing": False,
    }

    try:
        trainer = SFTTrainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    except TypeError:
        # Older TRL versions use tokenizer= instead of processing_class=.
        trainer = SFTTrainer(
            **trainer_kwargs,
            tokenizer=tokenizer,
        )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


def load_model_with_adapter(config: Mapping[str, Any]):
    """
    Load the base model and attach a saved LoRA adapter for evaluation.

    Adapters are small and portable; the base model is still loaded from Hugging
    Face, then PEFT layers apply the task-specific fine-tuned weights.
    """

    from peft import PeftModel

    model, tokenizer = load_base_model_and_tokenizer(config)
    model = PeftModel.from_pretrained(model, config["adapter_output_dir"])
    return model, tokenizer


def compare_before_after(
    prompts: List[str],
    baseline_responses: List[str],
    fine_tuned_responses: List[str],
) -> List[JSONDict]:
    """Create structured before/after qualitative evaluation rows."""

    rows = []
    for prompt, baseline, tuned in zip(
        prompts,
        baseline_responses,
        fine_tuned_responses,
    ):
        rows.append(
            GenerationRecord(
                prompt=prompt,
                baseline_response=baseline,
                fine_tuned_response=tuned,
                notes=(
                    "Review clarity, task focus, formatting, and whether the "
                    "fine-tuned response better matches the dataset style."
                ),
            ).to_dict()
        )
    return rows


def save_evaluation_results(
    rows: List[Mapping[str, Any]],
    path: Path = EVALUATION_DIR / "before_vs_after_results.json",
) -> Path:
    """Save qualitative evaluation results for reproducibility."""

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "qualitative_before_after",
        "results": list(rows),
    }
    return save_json(path, payload)


def push_adapter_to_hub(
    model,
    tokenizer,
    repo_id: str,
    *,
    private: bool = True,
) -> None:
    """
    Optionally upload the LoRA adapter to Hugging Face Hub.

    This expects the user to authenticate first, for example with
    huggingface_hub.notebook_login(). The adapter is uploaded, not the full
    base model, which keeps the artifact small.
    """

    model.push_to_hub(repo_id, private=private)
    tokenizer.push_to_hub(repo_id, private=private)


def initialize_task2_artifacts() -> JSONDict:
    """
    Create the default config, dataset, and evaluation prompts.

    This is safe to run at the top of the notebook and makes the workflow
    reproducible from a clean clone.
    """

    ensure_task2_directories()
    config = resolve_config_paths(export_default_training_config())
    dataset = create_instruction_dataset(Path(config["dataset_path"]))
    prompts = save_sample_prompts(Path(config["sample_prompts_path"]))
    return {
        "config": config,
        "dataset_size": len(dataset),
        "num_eval_prompts": len(prompts),
    }
