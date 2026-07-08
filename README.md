# 🏋️ LLM Fitness Coach

**Final Year Project** — An AI-powered fitness coaching system that uses **Large Language Models (LLM)** and **Retrieval-Augmented Generation (RAG)** to generate personalized, evidence-based workout plans.

## 📖 Overview

This project combines the reasoning capabilities of an LLM with a curated knowledge base of sports science research (hypertrophy, periodization, recovery, injury prevention, nutrition, etc.) to provide fitness recommendations grounded in real studies — rather than relying purely on the model's general knowledge.

## ✨ Features

- 🤖 LLM-powered conversational fitness coaching
- 📚 RAG pipeline referencing peer-reviewed research and guidelines
- 🧍 Personalized workout plans based on user profile and goals
- 📊 Built on a knowledge base covering training splits, progressive overload, recovery, and injury-related research

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — web app interface
- **LLM API** (e.g. OpenAI / Anthropic)
- **RAG** — vector search over a custom knowledge base (`fitcoach_data/`)

## 🚀 Getting Started

1. Clone the repo:
```bash
   git clone https://github.com/mierdanial/fitness-llm-coach.git
   cd fitness-llm-coach
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Add your API key to `.streamlit/secrets.toml`:
```toml
   OPENAI_API_KEY = "your-key-here"
```

4. Run the app:
```bash
   streamlit run AIFIT.py
```

## 📁 Project Structure
fitness-llm-coach/
├── AIFIT.py              # Main Streamlit app
├── profile.json          # User profile data
├── plan.txt              # Generated workout plans
├── requirements.txt      # Python dependencies
└── fitcoach_data/        # RAG knowledge base (research articles, embeddings)

## 📌 Status

Actively in development as part of a Final Year Project.

## 📄 License

This project is for academic purposes as part of a Final Year Project.
