# LLM Fitness Coach

This is my Final Year Project. It's an AI fitness coach that uses an LLM combined with RAG (Retrieval-Augmented Generation) to give workout plan recommendations based on actual sports science research instead of just the model guessing.

## What it does

Basically you chat with it about your fitness goals and it generates a personalized workout plan. Instead of just relying on what the LLM already knows, it pulls from a set of research papers and articles I collected (things like hypertrophy, progressive overload, recovery, injury prevention, nutrition for athletes etc) so the recommendations are backed by actual studies.

## Tech used

- Python
- Streamlit for the web app
- LLM API for the chat/generation part
- RAG pipeline to search through the knowledge base before generating a response

## How to run it
git clone https://github.com/mierdanial/fitness-llm-coach.git
cd fitness-llm-coach
pip install -r requirements.txt
streamlit run AIFIT.py

You'll need an API key from [OpenRouter](https://openrouter.ai) — sign up there, get your key, and add it in `.streamlit/secrets.toml`:
OPENROUTER_API_KEY = "your-key-here"

## Files

- `AIFIT.py` - main app
- `profile.json` - stores user profile info
- `plan.txt` - saved workout plans
- `fitcoach_data/` - the research articles and RAG index used for retrieval
- `requirements.txt` - dependencies

## Notes

Still working on this, some parts might change as I keep testing/improving it. This is part of my FYP so it's still a work in progress.
