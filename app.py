import gradio as gr
import os
import pandas as pd
from groq import Groq
from duckduckgo_search import DDGS
from newspaper import Article
import requests

# 1. Initialize Groq Client
# Ensure GROQ_API_KEY is set in Hugging Face Secrets
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def fetch_content(input_val):
    """Detects if input is a URL or plain text and extracts data."""
    if input_val.startswith(("http://", "https://")):
        try:
            article = Article(input_val)
            article.download()
            article.parse()
            return f"Title: {article.title}\n\nContent: {article.text[:2000]}"
        except Exception as e:
            return f"Error reading URL: {str(e)}"
    return input_val

def advanced_fact_check(user_input, history):
    if not user_input.strip():
        return "Please provide a claim or link.", [], pd.DataFrame(history), 0

    # Step A: Content Extraction
    content =
