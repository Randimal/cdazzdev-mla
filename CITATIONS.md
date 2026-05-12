# CITATIONS.md

# Citations and External Resources

This repository uses a combination of open-source libraries, APIs, pretrained models, and development tools across all tasks.

The following resources were used during implementation.

---

# Core Python Ecosystem

- Python
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- pydantic

---

# Financial Data and Search APIs

## Yahoo Finance / yfinance

Used for:
- historical stock market data
- financial news retrieval
- volatility analysis inputs

Repository:
https://github.com/ranaroussi/yfinance

---

## DuckDuckGo Search

Used for:
- lightweight web search retrieval inside the agentic workflow

Package:
https://pypi.org/project/ddgs/

---

# LLM APIs and Inference

## Groq API

Used for:
- LLM-based sentiment analysis
- lightweight structured inference

Website:
https://groq.com/

Documentation:
https://console.groq.com/docs

---

# Hugging Face Ecosystem

## Transformers

Used for:
- model loading
- inference
- tokenization
- fine-tuning workflows

Repository:
https://github.com/huggingface/transformers

---

## PEFT

Used for:
- parameter-efficient fine-tuning
- LoRA adapter workflows

Repository:
https://github.com/huggingface/peft

---

## TRL

Used for:
- supervised fine-tuning workflows
- QLoRA training orchestration

Repository:
https://github.com/huggingface/trl

---

## BitsAndBytes

Used for:
- 4-bit quantization
- memory-efficient model loading

Repository:
https://github.com/bitsandbytes-foundation/bitsandbytes

---

## Datasets

Used for:
- dataset preparation
- JSONL dataset loading

Repository:
https://github.com/huggingface/datasets

---

## Hugging Face Hub

Used for:
- adapter publishing
- model artifact management

Website:
https://huggingface.co/

---

# Base Model

## Qwen/Qwen2.5-3B-Instruct

Used as the base model for Task 2 QLoRA fine-tuning.

Model:
https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

---

# AI-Assisted Development Tools

The implementation process also involved AI-assisted engineering tools including:

- ChatGPT
- Codex
- Gemini CLI

These tools were primarily used for:
- rapid iteration
- debugging assistance
- dependency troubleshooting
- implementation validation
- workflow refinement

All workflows, debugging, architectural decisions, and final repository integration were manually reviewed and validated.

---

# Notes

This repository is intended for:
- educational purposes
- workflow engineering demonstrations
- and technical assessment evaluation.

It is not intended for:
- real financial trading
- production investment advice
- or large-scale production model training.