import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from duckduckgo_search import DDGS

# Load a powerful reasoning model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Get the main text from the page
        return " ".join([p.text for p in soup.find_all('p')[:5]]) # Get first 5 paragraphs
    except Exception as e:
        return f"Error reading URL: {str(e)}"

def advanced_checker(user_input, history):
    # 1. Handle URL vs Text
    if user_input.startswith("http"):
        claim = get_url_content(user_input)
        display_name = "URL Content"
    else:
        claim = user_input
        display_name = claim

    # 2. Web Search for Evidence
    with DDGS() as ddgs:
        search_results = [r['body'] for r in ddgs.text(claim[:200], max_results=3)]
    
    evidence = " ".join(search_results) if search_results else "No external evidence found."

    # 3. AI Reasoning (Zero-Shot)
    # We ask the AI to decide if the claim is supported or refuted by the evidence
    labels = ["supported", "refuted", "neutral"]
    result = classifier(f"Claim: {claim} Evidence: {evidence}", candidate_labels=labels)
    
    verdict_map = {
        "supported": "✅ VERIFIED TRUE",
        "refuted": "❌ DISPROVED/FALSE",
        "neutral": "🤔 UNCERTAIN / NO DATA"
    }
    top_label = result['labels'][0]
    confidence = result['scores'][0]
    verdict = verdict_map[top_label]

    # 4. Update History
    new_entry = {"Source": display_name[:30] + "...", "Verdict": verdict, "Confidence": f"{confidence:.1%}"}
    history.append(new_entry)
    
    reasoning = f"Analysis: The AI found that this claim is {top_label}. Confidence: {confidence:.2%}.\n\nSource Excerpt: {evidence[:300]}..."
    
    return verdict, reasoning, pd.DataFrame(history), history

# --- Layout ---
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# 🚀 Pro AI Fact-Checker")
    gr.Markdown("Paste a **URL** or type a **Statement** to verify it against live web data.")
    
    state_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(label="Claim or Link", placeholder="https://news-site.com/article or 'The sky is green'")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            out_verdict = gr.Label(label="Verdict")
            
    out_reasoning = gr.Textbox(label="Detailed Reasoning", lines=5)
    
    gr.Markdown("### 📊 Session Activity")
    history_table = gr.Dataframe(headers=["Source", "Verdict", "Confidence"])

    btn.click(advanced_checker, [input_box, state_history], [out_verdict, out_reasoning, history_table, state_history])

if __name__ == "__main__":
    demo.launch()
