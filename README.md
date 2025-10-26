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
  2. Retrieve relevant info.  
  3. Integrate retrieved knowledge into its response generation.
- Keeps BLUE’s knowledge base **fresh and contextually aware**.

### 🧩 Configurable and Extensible - Default model: llama-3.2-3B-instruct - Swap configurations and weights easily:
```bash
 CONFIGURATIONS = { "DIM": 3072, "FFN_DIM": 8192, "N_LAYERS": 28, "N_HEADS": 24, "N_KV_HEADS": 8, "VOCAB_SIZE": 128256, "NORM_EPS": 1e-5, "ROPE_THETA": 500000, "MAX_BATCH_SIZE": 4, "MAX_SEQ_LEN": 6000, "N_KV_HEAD_REP": 24 // 8, "HEAD_DIM": 128 }
