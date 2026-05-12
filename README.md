# CDAZZDEV Senior Machine Learning Engineer Assessment

This repository contains my submission for the CDAZZDEV Senior Machine Learning Engineer technical assessment.

The project focuses on practical AI systems engineering rather than pure research experimentation. Across all tasks, the main priorities were:

- modular architecture
- observability
- structured outputs
- reproducibility
- graceful degradation
- notebook-friendly workflows
- practical deployment-oriented engineering

The repository is organized into three independent but related workflows covering:
- financial AI analysis
- parameter-efficient LLM fine-tuning
- observable agentic orchestration systems

---

# Repository Structure

```text
CDAZZDEV-MLE-Poshitha/
│
├── README.md
├── CITATIONS.md
├── REFLECTION.md
├── requirements.txt
├── .env.example
│
├── task1_financial/
├── task2_genai/
└── task3_agentic/
```

---

# Task Overview

## Task 3 — Agentic Financial Research Workflow

Task 3 implements an observable multi-agent financial research workflow focused on:
- financial analysis
- sentiment analysis
- agent orchestration
- structured outputs
- workflow memory
- execution tracing

The architecture intentionally favors explicit orchestration and explainability over highly autonomous black-box agent systems.

### Key Features

- Multi-agent orchestration
- Structured Pydantic validation
- JSONL execution tracing
- Persistent memory caching
- Follow-up question handling
- Graceful fallback behavior
- Modular prompt engineering

### Architecture

```text
Tools
  ↓
Quantitative Analyst Agent
  ↓
Sentiment Research Agent
  ↓
Risk Review Agent
  ↓
Portfolio Strategist Agent
  ↓
Memory Cache
```

---

## Task 1 — Financial AI Analysis Workflow

Task 1 implements a modular notebook-oriented financial analysis workflow.

The workflow combines:
- market data retrieval
- technical indicator analysis
- volatility estimation
- financial news retrieval
- sentiment analysis
- structured investment outlook generation

Several reusable components from Task 3 were intentionally reused to maintain consistent observability and structured-output handling across the repository.

### Key Features

- RSI analysis
- MACD analysis
- Bollinger Bands
- Annualized volatility estimation
- News retrieval
- LLM sentiment analysis
- Structured JSON outputs
- Visualization generation

---

## Task 2 — Lightweight QLoRA Fine-Tuning Workflow

Task 2 implements a practical QLoRA fine-tuning workflow using:
- Hugging Face Transformers
- PEFT
- TRL
- BitsAndBytes quantization

The workflow demonstrates parameter-efficient fine-tuning using LoRA adapters on limited hardware.

The focus was not large-scale benchmark optimization, but rather:
- reproducibility
- workflow clarity
- practical engineering
- realistic resource constraints

### Key Features

- QLoRA fine-tuning
- 4-bit quantization
- LoRA adapters
- Before/after evaluation
- Reproducible training configuration
- Hugging Face adapter export
- Colab-friendly workflow design

---

# Engineering Philosophy

A major goal of this repository was keeping the workflows:
- explainable
- modular
- observable
- interview-friendly

The implementations intentionally avoid unnecessary abstraction layers and overengineered autonomous systems.

Instead, the workflows focus on:
- deterministic orchestration
- validated structured outputs
- reusable tooling
- practical debugging
- realistic operational behavior

---

# Technologies Used

## Core Python Ecosystem

- Python
- Jupyter Notebook
- Pydantic
- pandas
- numpy
- matplotlib

## Financial & Data APIs

- yfinance
- DuckDuckGo Search
- Groq API

## LLM & Fine-Tuning Stack

- transformers
- peft
- trl
- bitsandbytes
- datasets
- Hugging Face Hub

---

# Running the Repository

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment variables:

```bash
cp .env.example .env
```

Add required API keys:

```env
GROQ_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

Then open the notebooks inside:
- `task1_financial/`
- `task2_genai/`
- `task3_agentic/`

and run the workflows step-by-step.

---

# Notes on AI-Assisted Development

Modern AI engineering increasingly involves AI-assisted workflows.

During implementation, I used tools such as:
- ChatGPT
- Codex
- Gemini CLI

primarily for:
- rapid iteration
- debugging
- environment troubleshooting
- implementation validation

However, the primary focus throughout the project was understanding, validating, debugging, and refining the workflows and engineering decisions rather than blindly generating code.

---

# Video Walkthrough

A short walkthrough video is included with the submission explaining:
- repository architecture
- workflow design
- engineering decisions
- observability strategy
- structured output handling
- implementation tradeoffs

---

# Final Notes

This repository was designed to demonstrate practical AI engineering skills across:
- workflow orchestration
- financial AI systems
- LLM fine-tuning
- observability
- structured validation
- production-oriented software engineering

The emphasis throughout the project was building understandable, reproducible, and maintainable systems rather than overly complex research prototypes.