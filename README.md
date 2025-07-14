# SAGE

**SAGE: Data-Semantics-Aware Recommendation of Diverse Pivot Tables**

This repository contains the implementation of **SAGE**, Data-Semantics-Aware Recommendation of Diverse Pivot Tables. 

---

## 📦 Requirements

First, set up your Python environment and install the dependencies.

### Use `requirements.txt`
```
pip install -r requirements.txt
```

---

## 🤗 Hugging Face Model Access
SAGE uses models like [LLaMA](https://huggingface.co/meta-llama/Llama-3.1-8B),
TAPEX, and T5 hosted on Hugging Face. Access to these models may be restricted.

Steps:
1. Request access to the model by following the link.
2. Once approved, create a Hugging Face token from your settings page.
3. Save your token in a file named `hf_token.txt` in the root directory.

---

## 🚀 How to Run
After setting up dependencies and your token, you can run the main script
```
bash run_sage.sh
```

---

## 📂 Project Structure
├── hf_token.txt             # Hugging Face access token
├── requirements.txt         # Python dependencies
├── run_sage.sh              # The script to run the SAGE algorithm
├── src/                     # Source code modules
├── dataset/                 # Input datasets 
├── precompute/              # Precomputed models and results
└── README.md