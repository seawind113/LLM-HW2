# LLM Homework 2: Text Summarization and RAG Systems

This assignment is part of the coursework for the Large Language Model course. It includes implementations and experiments on **Text Summarization** using T5-small and **Retrieval-Augmented Generation (RAG)** systems using LangChain and LLaMA3.2 3B. The work is split into two main parts as per the assignment instructions.

## Table of Contents

- [Overview](#overview)
- [Part 1: Text Summarization](#part-1-text-summarization)
  - [1.1 Fine-tuning T5-small](#11-fine-tuning-t5-small)
  - [1.2 Evaluation via Custom ROUGE Metrics](#12-evaluation-via-custom-rouge-metrics)
  - [1.3 Zero-shot Summarization with Hard Prompt](#13-zero-shot-summarization-with-hard-prompt)
  - [1.4 Soft Prompt Tuning](#14-soft-prompt-tuning)
  - [1.5 Comparison and Discussion](#15-comparison-and-discussion)
- [Part 2: Retrieval-Augmented Generation](#part-2-retrieval-augmented-generation)
  - [2.1 Standard RAG Pipeline](#21-standard-rag-pipeline)
  - [2.2 Multi-Query RAG](#22-multi-query-rag)
  - [2.3 RAG Fusion with Reciprocal Rank Fusion](#23-rag-fusion-with-reciprocal-rank-fusion)
- [File Structure](#file-structure)
- [License](#license)

---

## Overview

This repository contains my solutions for LLM Homework 2, covering both text summarization tasks and retrieval-augmented generation (RAG). The solutions include:

- T5-small fine-tuning for summarization using CNN/DailyMail dataset
- Implementation of ROUGE-1, ROUGE-2, and ROUGE-L from scratch
- Prompt-based zero-shot summarization
- Soft prompt tuning without full retraining
- RAG with LangChain, Multi-Query RAG, and RAG Fusion using Reciprocal Rank Fusion

## Part 1: Text Summarization

### 1.1 Fine-tuning T5-small

- Dataset: CNN/DailyMail with 2900+ training samples
- Model: `google-t5/t5-small` (Hugging Face)
- Input: `article`
- Target: `highlight`
- No prompt is added during training

📈 Output: Learning curves for both training and validation loss are provided.

### 1.2 Evaluation via Custom ROUGE Metrics

Implemented ROUGE-1, ROUGE-2, and ROUGE-L **from scratch**, without using any external library like `load_metric`. Evaluation is performed on the `test.csv` dataset.

### 1.3 Zero-shot Summarization with Hard Prompt

- Format: `summarize: {input}`
- Model: Pretrained T5-small (no fine-tuning)
- Evaluation: Custom ROUGE metrics

### 1.4 Soft Prompt Tuning

- Uses prompt embeddings prepended to inputs
- Only the prompt embeddings are updated (model remains frozen)
- Evaluated using custom ROUGE metrics
- 📈 Learning curves included

### 1.5 Comparison and Discussion

A comparison of all three summarization methods (fine-tuning, hard prompts, and soft prompts), including discussion on performance, strengths, and limitations. The ROGUE-scores exceeds the initial baseline (Average ROGUE-1: 0.3615, Average ROGUE-2: 0.1485, Average ROGUE-1: 0.2471), 
demonstrating the competence of the training results. 

## Part 2: Retrieval-Augmented Generation

### 2.1 Standard RAG Pipeline

- Built using LangChain
- Components: Retriever, Prompt, LLM (LLaMA3.2 3B), and Output Parser
- 🔍 Includes discussion on TextSplitter parameters and Retriever settings

### 2.2 Multi-Query RAG

- Generates multiple sub-queries from one question
- Chains them through the RAG pipeline
- 🔄 Compares performance with single-query RAG

### 2.3 RAG Fusion with Reciprocal Rank Fusion

- Implements RRF for combining ranked results from multiple sub-queries
- Discusses the effect of the `c` parameter on output relevance
- 🚀 Fusion chain built similarly to standard RAG chain

