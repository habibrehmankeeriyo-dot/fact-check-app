import gradio as gr
import os
import pandas as pd
from groq import Groq
from duckduckgo_search import DDGS

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def advanced_fact_check(claim, history):
    if not claim.strip():
        return "Please enter a claim.", [], pd.DataFrame(history)

    # 1. Fetch search results with URLs
    sources = []
    with DDGS() as ddgs:
        results = list(ddgs.text(claim, max_results=4))
        search_text = ""
        for r in results:
            search_text += f"Source: {r['title']} - {r['body']}\n"
            sources.append([r['title'], r['href']])
    
    # 2. Reasoning with Llama 3.3
    prompt = f"""
    Analyze this claim based on the provided evidence. 
    1. Direct Verdict (TRUE/FALSE/UNCERTAIN)
    2. One-sentence reasoning.
    3. Key evidence summary.
    
    Claim: {claim}
    Evidence: {search_text}
    """

    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a senior investigative journalist. Be precise and skeptical."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    analysis = chat.choices[0].message.content
    verdict = "TRUE" if "TRUE" in analysis.upper() else "FALSE" if "FALSE" in analysis.upper() else "UNCERTAIN"
    
    # 3. Update History
    history.append({"Claim": claim[:30] + "...", "Verdict": verdict})
    
    return analysis, sources, pd.DataFrame(history)

# --- Pro UI Layout ---
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# 🛡️ Investigative AI Fact-Checker")
    
    state_history = gr.State([])

    with gr.Tabs():
        with gr.TabItem("🔍 Analyze"):
            with gr.Row():
                with gr.Column(scale=3):
                    claim_input = gr.Textbox(label="Claim", placeholder="Enter a statement to verify...")
                    check_btn = gr.Button("Run Audit", variant="primary")
                with gr.Column(scale=1):
                    # Shows a color-coded label
                    verdict_label = gr.Label(label="Quick Verdict")
            
            output_report = gr.Markdown(label="Full Investigation Report")
            
        with gr.TabItem("🌐 Sources"):
            gr.Markdown("### Evidence gathered from the web:")
            source_display = gr.Dataframe(headers=["Title", "URL"], interactive=False)

        with gr.TabItem("📊 Session Log"):
            history_table = gr.Dataframe(headers=["Claim", "Verdict"])

    # Connection logic
    def update_label(analysis):
        if "TRUE" in analysis.upper(): return "TRUE"
        if "FALSE" in analysis.upper(): return "FALSE"
        return "UNCERTAIN"

    check_btn.click(
        fn=advanced_fact_check, 
        inputs=[claim_input, state_history], 
        outputs=[output_report, source_display, history_table]
    ).then(
        fn=lambda x: x.split('\n')[0], # Extracting first line for label
        inputs=[output_report],
        outputs=[verdict_label]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
