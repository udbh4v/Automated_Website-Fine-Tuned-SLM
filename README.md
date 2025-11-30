# General_Purpose_SLM

# Automated Website Dataset Builder & Fine-Tuned SLM 🚀

A complete pipeline to **automatically build structured datasets from any website** and **fine-tune a Small Language Model (SLM)** on that dataset.  
This enables you to quickly create domain-specific AI models based on real website content — efficient, lightweight, and cost-friendly.

---

## 🌐 Overview

This repository contains two core components:

1. **Dataset Builder (`BUILD_WEBSITE_DATASET.py`)**  
   Automatically crawls/extracts content from any website URL and converts it into a clean, structured dataset suitable for model training.

2. **Fine-Tuning Script (`slm.py`)**  
   Takes the generated dataset and fine-tunes a Small Language Model (SLM) to understand and generate content consistent with your website domain.

---

## 📁 Repository Structure.
├── BUILD_WEBSITE_DATASET.py # Automated website data extraction + dataset creation
├── slm.py # Fine-tuning pipeline for the SLM
├── README.md # Documentation
└── (additional config files if added later)


---

## ✨ Features

- ✔️ Fully automated website-to-dataset pipeline  
- ✔️ Preprocessing, cleaning, and structuring included  
- ✔️ Fine-tuning done on any small open-source model  
- ✔️ Efficient even on low compute  
- ✔️ Modular design — easy to extend or integrate into other projects  

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/udbh4v/Automated_Website-Fine-Tuned-SLM
cd Automated_Website-Fine-Tuned-SLM
pip install -r requirements.txt


# 📘 USER GUIDE  
## Automated Website Dataset Builder & Fine-Tuned SLM

Welcome to the official **User Guide** for the *Automated Website → Dataset → Fine-Tuned SLM* pipeline.  
This document explains how to install, run, customize, and extend the system — even if you have zero prior experience with model training.

---

# 📂 1. Introduction

This project helps you:

- Extract structured text data from any website  
- Clean and convert it into a machine-readable dataset  
- Fine-tune a Small Language Model (SLM) on that dataset  
- Build your own domain-specific AI model  

It is lightweight, efficient, and ideal for business websites, SaaS platforms, blogs, and documentation portals.

---

# ⚙️ 2. Prerequisites

Before beginning, ensure you have:

- Python **3.8+**
- Git installed  
- A machine with:
  - CPU (minimum)
  - GPU (recommended, e.g., CUDA-enabled)

Install dependencies:

```bash
pip install -r requirements.txt
