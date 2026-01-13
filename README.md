# GPT-Style Decoder-Only Transformer from Scratch

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Optimization: AdamW](https://img.shields.io/badge/Optimizer-AdamW-blue?style=flat-square)](https://arxiv.org/abs/1711.05101)
[![Status: Ongoing Research](https://img.shields.io/badge/Status-Ongoing%20Research-success?style=flat-square)](https://github.com/Owais-Ml-Dev/GPT-from-Scratch)

A first-principles implementation of a Generative Pre-trained Transformer (GPT) architecture. This repository bypasses high-level abstractions to manually engineer the internal mechanics of Large Language Models, focusing on tensor-level dynamics, training stability, and causal inference.

---

## 🎯 Engineering Objectives

In an era of pre-trained "black-box" models, this project serves as a deep-dive into the fundamental bottlenecks of transformer architectures. The focus is on:
- **Tensor-Level Causal Attention:** Implementing masked multi-head attention to ensure rigorous next-token prediction without "look-ahead" bias.
- **Architectural Stability:** Managing gradient flow through residual connections and Layer Normalization.
- **Inference Optimization:** Engineering sophisticated sampling strategies to balance coherence and creativity.
- **Optimization Intuition:** Monitoring loss convergence and weight updates to understand the behavior of the **AdamW** optimizer in high-dimensional spaces.

---

## 🏗️ Architectural Blueprint

The model is a decoder-only transformer designed for autoregressive language modeling.

### Core Components
* **Embeddings:** Dual-path input processing involving both **Token Embeddings** and **Learned Positional Embeddings** to retain spatial sequence information.
* **Transformer Blocks:** * **Multi-Head Causal Self-Attention:** Parallelized attention heads with scaled dot-product attention and triangular masking.
    * **Feed-Forward Networks (FFN):** Position-wise networks utilizing **GELU (Gaussian Error Linear Unit)** activations for non-linear feature mapping.
    * **Normalization:** Layer Normalization applied before the attention and FFN blocks (Pre-LN architecture) to improve training stability.
* **Projection Head:** A final linear layer mapping hidden states to the vocabulary dimension for probability distribution modeling.

---

## 📊 Data Engineering & Training

### Dataset Characteristics
- **Source:** Instruction-based dataset containing **40,000 samples**.
- **Preprocessing:** Custom PyTorch `Dataset` and `DataLoader` implementation with dynamic padding.
- **Tokenization:** GPT-2 compatible byte-pair encoding (BPE) using **tiktoken**.

### Training Hyperparameters & Logistics
| Hyperparameter | Value |
| :--- | :--- |
| **Objective** | Next-Token Prediction (Cross-Entropy Loss) |
| **Optimizer** | AdamW (Weight Decay Fix) |
| **Dataset Split** | 34k Train / 2k Val / 4k Test |
| **Convergence** | Monitored via Validation-based Early Stopping |

The pipeline includes **Shifted Target Handling**, ensuring the model learns to predict the $n+1$ token based solely on $1 \dots n$ tokens.

---

## ⚡ Inference & Generation Strategies

Moving beyond simple greedy decoding, I implemented diverse sampling strategies to control the model’s stochastic behavior:

* **Temperature-Based Sampling:** Adjusting the "softness" of the probability distribution to control randomness.
* **Top-K Sampling:** Filtering the top $K$ most likely next tokens to prune the long tail of low-probability distributions, significantly reducing "gibberish" generation.
* **End-of-Sequence (EOS) Handling:** Ensuring the model respects logical termination boundaries.

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch (Tensor manipulation, Autograd, Module API)
- **Scientific Computing:** NumPy, Matplotlib (Loss visualization)
- **Tokenization:** Tiktoken (OpenAI’s BPE implementation)
- **Environment:** GPU-accelerated training for efficient tensor operations

---

## 🚀 Research Roadmap (Next Steps)

- [ ] **PEFT Implementation:** Integrating **LoRA (Low-Rank Adaptation)** for efficient parameter fine-tuning.
- [ ] **Evaluation Metrics:** Implementing **Perplexity (PPL)** calculations on the test set.
- [ ] **Scaling:** Increasing context window length and model depth for complex instruction following.
- [ ] **Refactoring:** Transitioning to a highly modular, configuration-driven code structure (YAML-based).

---

## Author

**Owais Shaikh** *AI and Machine Learning Engineer* Specializing in LLM Internals, NLP Research, and Deep Learning Systems.

---

> **Note:** This project is a research-driven foundation intended for technical mastery of transformer dynamics and convergence behavior.
