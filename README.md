# GPT Model Built From Scratch (PyTorch)

🚧 **Ongoing** 🚧

This repository contains an end to end implementation of a GPT style decoder only transformer built entirely from scratch using PyTorch. The objective of this project is to deeply understand how large language models work internally, rather than relying on pretrained abstractions.

---

## Motivation

Most modern LLM workflows focus on fine tuning existing models. While powerful, this often hides the internal mechanics of transformers.

This project was built from first principles to understand:
- How causal self attention works at the tensor level
- How transformer blocks learn contextual relationships
- How loss behaves during large scale language modeling
- What affects training stability and convergence
- How instruction tuning shapes model behavior

Building every component manually has provided strong architectural and optimization level intuition behind LLMs.

---

## Implemented Features

### Model Architecture
- GPT style decoder only transformer
- Token embeddings and learned positional embeddings
- Multi head causal self attention with masking
- Transformer blocks with residual connections and layer normalization
- Feedforward networks with GELU activation
- Final projection head for next token prediction

### Tokenization
- GPT 2 compatible tokenization using tiktoken

## Dataset and Training Setup

- Instruction based JSON dataset with 40,000 samples
- Dataset split:
- Training: 34,000
- Validation: 2,000
- Test: 4,000
- Custom PyTorch Dataset and DataLoader
- Dynamic padding and attention mask handling
- Shifted targets for next token prediction

---

## Training Pipeline

- Objective: Next token prediction
- Loss function: Cross entropy
- Optimizer: AdamW
- Validation based early stopping
- Periodic evaluation during training
- GPU supported training

Training and validation loss show stable convergence with no major overfitting, aligning with expected GPT style learning behavior.

---

## Text Generation

- Greedy decoding
- Temperature based sampling
- Top k sampling
- Generation stops at end of text token

The model is already capable of generating coherent and context aware responses from instruction prompts.

🚧 **Currently in Progress**

Planned improvements:
- Longer training runs on larger datasets
- Perplexity based evaluation
- Instruction fine tuning experiments
- LoRA based parameter efficient tuning
- Improved sampling strategies
- Code refactoring and modularization
- Better experiment tracking and documentation

---

## Tech Stack

- Python
- PyTorch
- NumPy
- tiktoken
- Matplotlib

---

## Disclaimer

This project is built for learning and research purposes to gain a deep understanding of transformer based language models. It is not intended to compete with production scale models.

---

## Author

**Owais Shaikh**  
Aspiring AI and Machine Learning Engineer  
Focus Areas: Large Language Models, NLP, Deep Learning Systems

- GPT 2 compatible tokenization using tiktoken
- Instruction formatted prompts using:
