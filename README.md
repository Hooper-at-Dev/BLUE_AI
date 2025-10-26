# 🔷 BLUE AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu128-orange.svg)](https://pytorch.org/)
[![Model: LLaMA3](https://img.shields.io/badge/Model-LLAMA3-informational.svg)](https://ai.meta.com/llama/)
[![RAG Enabled](https://img.shields.io/badge/RAG-Enabled-success.svg)](#retrieval-augmented-generation)

> A scalable, modular reimplementation of the **LLaMA 3** family — from 1B to 405B parameters — with optional **web-based RAG** (Retrieval-Augmented Generation).

---

## 🚀 Overview

**BLUE AI** is a next-generation large language model, designed for scalability and adaptability across various model sizes of **LLAMA3**.

At its core, BLUE is a **reimplementation of Meta’s LLaMA 3 family** of text models — fully modular and designed to **scale effortlessly** from lightweight inference on consumer hardware to multi-GPU inference with large parameter sets.

BLUE also integrates a **web-based RAG system**, enabling it to **search the internet in real time**, retrieve relevant context, and use that knowledge to craft accurate and up-to-date responses.

---

## 🧠 Key Features

### 💡 Scalable Model Architecture
- Choose from **1B → 405B** parameter configurations.
- Switch between models with ease.
- Built with **efficiency, extensibility**, and **hardware flexibility** in mind.

### 🌐 Web-Based RAG (Retrieval-Augmented Generation)
- Optional RAG mode allows BLUE to:
  1. Query the web for real-time data.  
  2. Retrieve and rank relevant documents.  
  3. Integrate retrieved knowledge into its response generation.
- Keeps BLUE’s knowledge base **fresh and contextually aware**.

###🧩 Configurable and Extensible

BLUE is fully modular — users can easily switch between different model variants by modifying a simple Python configuration block.
The default configuration uses LLaMA-3.2-3B-Instruct.

# Example configuration for BLUE using LLaMA-3.2-3B-Instruct

CONFIGURATIONS = {
    "DIM": 3072,
    "FFN_DIM": 8192,
    "N_LAYERS": 28,
    "N_HEADS": 24,
    "N_KV_HEADS": 8,
    "VOCAB_SIZE": 128256,
    "NORM_EPS": 1e-5,
    "ROPE_THETA": 500000,
    "MAX_BATCH_SIZE": 4,
    "MAX_SEQ_LEN": 6000,
    "N_KV_HEAD_REP": 24 // 8,
    "HEAD_DIM": 128
}

# Update tokenizer and weight paths for the chosen model
tok_DIR = "Weights/3B-instruct/original/tokenizer.model"
weight_DIR = "./Weights/3B-instruct/original/consolidated.00.pth"


By adjusting these parameters and file paths, you can instantly switch between 1B, 3B, 70B, or even 405B model configurations — without altering the core logic of BLUE.
This design makes it easy to experiment, scale, and fine-tune models of varying sizes using the same unified interface.
