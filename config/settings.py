import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multimodal-ai-assistant")

if LANGCHAIN_TRACING_V2.lower() == "true" and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Anthropic models
ANTHROPIC_VISION_MODEL = "claude-sonnet-4-6"
ANTHROPIC_FAST_MODEL = "claude-haiku-4-5-20251001"

# Groq vision-capable models  (free tier)
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_FAST_MODEL = "llama-3.3-70b-versatile"

# Pricing per million tokens (USD)
MODEL_PRICING = {
    "claude-sonnet-4-6":                          {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001":                  {"input": 0.25,  "output": 1.25},
    "meta-llama/llama-4-scout-17b-16e-instruct":  {"input": 0.11,  "output": 0.34},
    "llama-3.3-70b-versatile":                    {"input": 0.59,  "output": 0.79},
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
BILLING_POLICIES_DIR = os.path.join(BASE_DIR, "data", "billing_policies")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
