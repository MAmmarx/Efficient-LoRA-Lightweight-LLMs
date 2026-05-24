# Efficient LoRA Fine-Tuning of Lightweight LLMs for Sentiment Classification

This project introduces a rigorous, performance-driven approach to parameter-efficient fine-tuning (PEFT). It optimizes lightweight Large Language Models (LLMs) for specialized text classification tasks by systematically evaluating foundational architectures under low-rank adaptation, bridging the gap between high-overhead full model fine-tuning and low-accuracy zero-shot baselines.

**Academic Research & Implementation | Accepted to IEEE (IMSA 2026)**

> This algorithm and the accompanying research paper were developed and published during my **3rd year of undergraduate studies**, representing an early-career contribution to the fields of Natural Language Processing and Computational Efficiency.

## Overview
Fully fine-tuning massive language models remains computationally prohibitive and memory intensive, especially in resource-constrained environments. This project implements an empirical evaluation framework utilizing **Low-Rank Adaptation (LoRA)** to optimize compact model backbones—specifically **BLOOM-560M**, **Qwen3.5-0.6B**, and **GPT-2**. 

By freezing over 98% of the pre-trained weight matrices and injecting trainable low-rank decomposition matrices into the multi-head attention layers, the system circumvents overfitting and training collapse. The framework was rigorously evaluated across two classification benchmarks—**Rotten Tomatoes** (general sentiment) and **Twitter Financial News** (domain-specific financial sentiment)—demonstrating robust accuracy gains exceeding 30 to 55 percentage points relative to zero-shot baselines.

## Features
- **Parameter-Efficient Scaling:** Drastically reduces memory usage by targeting updates to a tiny fraction of parameters, down to 0.14% on BLOOM-560M.
- **Zero-Latency Deployment:** LoRA adapter matrices merge directly into base model weights post-training, preserving original inference runtime speeds.
- **Multi-Backbone Framework:** Out-of-the-box integration with diverse architectural blocks, including GPT-2, BLOOM, and Qwen styles.
- **Domain Adaptation Pipeline:** Built-in loaders handle tokenization, structural imbalance mitigation, and dataset formatting for binary and ternary sentiment tasks.
- **Unified Hyperparameter Execution:** Configured using a standardized optimization setup (AdamW, fixed scheduling) to facilitate perfectly unbiased cross-model evaluation.

## How It Works
- **Base Matrix Freezing:** The pre-trained weights ($W \in \mathbb{R}^{d \times k}$) are strictly frozen to preserve foundational language knowledge.
- **Low-Rank Matrix Injection:** Two low-dimensional matrices, $A$ and $B$, are injected into the Query and Value projection sub-layers of the attention module.
- **Forward Propagation Update:** The overall weight modification is parameterized as $\Delta W = B \cdot A$, where rank $r \ll \min(d, k)$.
- **Scaling Correction:** An isolated scaling factory ($\alpha$) rescales the adapter updates to regulate contributions smoothly against base weight magnitudes.
- **Evaluation Convergence:** Tracking metrics calculate micro/macro precision, recall, and harmonic F1-scores across testing sets.

## Research Publication
This implementation is based on the paper: 
**"Efficient LoRA Fine-Tuning of lightweight LLMs for Sentiment Classification"** *Authors: Mohamed Alkhowaiter & Mohammed Ammar (Accepted, IMSA 2026 / IEEE)*

## Prerequisites
- Python 3.x
- PyTorch
- Hugging Face `transformers`, `peft`, `datasets`, and `accelerate`
- scikit-learn & pandas
