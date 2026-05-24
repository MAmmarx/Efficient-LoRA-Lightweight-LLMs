# Efficient LoRA Fine-Tuning of Lightweight LLMs for Sentiment Classification

[cite_start]This project introduces a rigorous, performance-driven approach to parameter-efficient fine-tuning (PEFT)[cite: 8, 26]. [cite_start]It optimizes lightweight Large Language Models (LLMs) for specialized text classification tasks by systematically evaluating foundational architectures under low-rank adaptation, bridging the gap between high-overhead full model fine-tuning and low-accuracy zero-shot baselines[cite: 6, 7, 8].

**Academic Research & Implementation | [cite_start]Accepted to IEEE (IMSA 2026)** 

> This algorithm and the accompanying research paper were developed and published during my **3rd year of undergraduate studies**, representing an early-career contribution to the fields of Natural Language Processing and Computational Efficiency.

## Overview
[cite_start]Fully fine-tuning massive language models remains computationally prohibitive and memory intensive, especially in resource-constrained environments[cite: 16]. [cite_start]This project implements an empirical evaluation framework utilizing **Low-Rank Adaptation (LoRA)** to optimize compact model backbones—specifically **BLOOM-560M**, **Qwen3.5-0.6B**, and **GPT-2**. 

[cite_start]By freezing over 98% of the pre-trained weight matrices and injecting trainable low-rank decomposition matrices into the multi-head attention layers [cite: 32, 36][cite_start], the system circumvents overfitting and training collapse[cite: 6, 58]. [cite_start]The framework was rigorously evaluated across two classification benchmarks—**Rotten Tomatoes** (general sentiment) and **Twitter Financial News** (domain-specific financial sentiment) [cite: 9, 106][cite_start]—demonstrating robust accuracy gains exceeding 30 to 55 percentage points relative to zero-shot baselines[cite: 10].

## Features
- [cite_start]**Parameter-Efficient Scaling:** Drastically reduces memory usage by targeting updates to a tiny fraction of parameters, down to 0.14% on BLOOM-560M[cite: 12, 13, 73].
- [cite_start]**Zero-Latency Deployment:** LoRA adapter matrices merge directly into base model weights post-training, preserving original inference runtime speeds[cite: 63, 77].
- [cite_start]**Multi-Backbone Framework:** Out-of-the-box integration with diverse architectural blocks, including GPT-2, BLOOM, and Qwen styles[cite: 9].
- [cite_start]**Domain Adaptation Pipeline:** Built-in loaders handle tokenization, structural imbalance mitigation, and dataset formatting for binary and ternary sentiment tasks[cite: 108, 113, 124].
- [cite_start]**Unified Hyperparameter Execution:** Configured using a standardized optimization setup (AdamW, fixed scheduling) to facilitate perfectly unbiased cross-model evaluation[cite: 126, 127].

## How It Works
- [cite_start]**Base Matrix Freezing:** The pre-trained weights ($W \in \mathbb{R}^{d \times k}$) are strictly frozen to preserve foundational language knowledge[cite: 61, 67].
- [cite_start]**Low-Rank Matrix Injection:** Two low-dimensional matrices, $A$ and $B$, are injected into the Query and Value projection sub-layers of the attention module[cite: 70, 99].
- [cite_start]**Forward Propagation Update:** The overall weight modification is parameterized as $\Delta W = B \cdot A$, where rank $r \ll \min(d, k)$[cite: 68, 71].
- [cite_start]**Scaling Correction:** An isolated scaling factory ($\alpha$) rescales the adapter updates to regulate contributions smoothly against base weight magnitudes[cite: 103].
- [cite_start]**Evaluation Convergence:** Tracking metrics calculate micro/macro precision, recall, and harmonic F1-scores across testing sets[cite: 121, 140, 163].

## Research Publication
This implementation is based on the paper: 
[cite_start]**"Efficient LoRA Fine-Tuning of lightweight LLMs for Sentiment Classification"** *Authors: Mohamed Alkhowaiter & Mohammed Ammar (Accepted, IMSA 2026 / IEEE)* [cite: 1, 2, 3, 5]

## Prerequisites
- Python 3.x
- [cite_start]PyTorch [cite: 124]
- [cite_start]Hugging Face `transformers`, `peft`, `datasets`, and `accelerate` [cite: 124]
- scikit-learn & pandas
