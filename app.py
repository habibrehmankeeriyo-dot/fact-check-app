import gradio as gr
import os
import pandas as pd
from groq import Groq
from duckduckgo_search import DDGS

# 1. Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def fact_checker(claim, history):
    if not claim.strip():
        return "Please enter a claim.", history, pd.DataFrame(history)

    # 2. Search for Evidence
    with DDGS() as ddgs:
        search_results = [r['body'] for r in ddgs.text(claim, max_results=3)]
    
    evidence = "\n".join(search_results) if search_results else "No online evidence found."

    # 3. Use Groq LLM for Reasoning
    # We give the LLM the claim and the evidence we found
    prompt = f"""
    You are a professional fact-checker. 
    Claim: {claim}
    Evidence: {evidence}
    
    Based ONLY on the evidence provided, determine if the claim is TRUE, FALSE, or UNCERTAIN.
    Provide a short explanation (max 3 sentences).
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    
    analysis = chat_completion.choices[0].message.content
    
    # 4. Determine Verdict for History Table
    verdict = "TRUE" if "TRUE" in analysis.upper() else "FALSE" if "FALSE" in analysis.upper() else "UNCERTAIN"
    
    # Update History
    history.append({"Claim": claim[:50] + "...", "Verdict": verdict})
    
    return analysis, history, pd.DataFrame(history)

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# ⚡ Groq-Powered Ultra Fast Fact Checker")
    state_history = gr.State([])

    with gr.Row():
        input_box = gr.Textbox(label="Enter Statement", placeholder="e.g. Is the coffee good for health?")
        btn = gr.Button("Fact Check", variant="primary")
    
    output_text = gr.Textbox(label="AI Analysis (Powered by Groq)")
    
    gr.Markdown("### 🕒 Session History")
    history_table = gr.Dataframe(headers=["Claim", "Verdict"])

    btn.click(fact_checker, [input_box, state_history], [output_text, state_history, history_table])

if __name__ == "__main__":
    demo.launch()
