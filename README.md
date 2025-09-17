# Humor_LLM

A system for generating Portuguese humor using local **Ollama** models.

---

## Table of Contents

- [About](#about)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Notes](#notes)  

---

## About

This repository provides code to generate humorous text in Portuguese using **LLMs running locally via Ollama**. Unlike cloud-based APIs, Ollama models are stored and executed on your machine, so no external calls are required.  

---

## Prerequisites

- Python (≥3.10 recommended)  
- Ollama installed and configured locally ([https://ollama.com](https://ollama.com))  
- Local Ollama models downloaded (check available models with `ollama list`)  
- `openai` Python package (for interfacing with Ollama in your scripts)  

---

## Installation

1. Clone the repository:

```bin
git clone https://github.com/andrefmoreira/Humor_LLM
cd Humor_LLM
```

2. Create and activate a virtual environment:

```bin
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # macOS/Linux
````

3. Install dependencies:

```bin
pip install -r requirements.txt
```

## Usage

1. Ensure Ollama is running locally and the model you want to use is available.
2. Run the create_dataset.py
3. Run the either few_shot.py or zero_shot.py.
4. Optionally, if you want to run the model on the given headlines, make sure you have the headlines.json file and run the few_shot_news.py or the zero_shot_news.py

## Notes

The repository does not provide pre-trained models; you must have your Ollama models installed locally.
Since models run locally, GPU/CPU performance will affect generation speed.
Ensure Python scripts point to the correct local Ollama model paths.
