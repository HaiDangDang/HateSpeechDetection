# Hate Speech Detection Pipeline

## 📌 Overview
This repository contains a full pipeline for hate speech detection, from data collection to model evaluation. The project includes multiple filtering and labeling steps, leveraging LLMs and LightGBM models to annotate and classify hate speech.

## 📂 Project Structure

### 1️⃣ `1_collection/`  
- Code for collecting data from OpenWebSearch using two different methods:
  - **Method 1**: Collects plain text and applies additional noise filtering.
  - **Method 2**: Extracts structured discussion content matching Schema.org format.

### 2️⃣ `2_filtering/`  
- **Faiss Filtering**: Applies FAISS-based similarity search for deduplication and quality filtering.
- **LLM-Based Filtering**: Uses large language models (LLMs) to refine and filter collected data.

### 3️⃣ `3_labeling/`  
- Generates **probability values** for classification labels.
- Uses multiple models to assign a probabilistic score to each sample.

### 4️⃣ `4_lgb/`  
- Builds a **LightGBM (LGB) model** for hate speech classification.
- Features extracted from annotated data to improve classification performance.

### 5️⃣ `5_train_annotation/`  
- Training pipeline includes **four ensemble models**.
- Fine-tunes models using **LoRA (Low-Rank Adaptation)**.

### 6️⃣ `6_evaluation/`  
- **Evaluation of binary and multi-class hate speech detection tasks**.
- Runs inference using LLM endpoints:
  - **LLaMA 70B**
  - **Gemma2 27B**
- Includes full results of all models.
- `all_binary_result/` contains probability values for:
  - **Label 1** (Hate Speech)
  - **Label 2** (Normal Speech)

## 📁 Data Structure

### `data/`
- Raw text collected from OpenWebSearch.
- Processed speech/text after filtering.

### `human_data/`
- Human-labeled datasets.
- Contains both binary and multi-class labeled datasets.

### `binary/` and `multi/`
- Annotated datasets with **probability values** assigned by four models:
  - **Qwen2.5 14B**
  - **LLaMA3.1 8B**
  - **Mistral 7B**
  - **Gemma2 9B**

## 🚀 LoRA Fine-Tuned Models on Hugging Face

### **Binary Classification Models**
- `dangdangde/m2.v2.lgb.llama1B`
- `dangdangde/m2.v2.lgb.Qwen14B`
- `dangdangde/m2.v8.lgb.Llama1B`
- `dangdangde/m2.v9.lgb.Llama1B`
- `dangdangde/m1.m2.v1.mean.Llama1B`
- `dangdangde/m2.v4.lgb.Qwen14B`

### **Multi-Class Classification Models**
- `dangdangde/multi.lgb.Llama1B`
- `dangdangde/m2.multi.v3.mean.Llama1B`
- `dangdangde/multi.lgb.Qwen14B`
- `dangdangde/m2.multi.v2.mean.Qwen14B`

These models are available on **Hugging Face** for fine-tuned hate speech detection.


## 💾 Large File Storage
For large files related to this project, you can access them via **Google Drive**:
[Large File Storage](https://drive.google.com/drive/folders/1LL21xuq3LcupI7nOYE6EDpiA_XGiTJrj?usp=drive_link)


