import gradio as gr
import pandas as pd
from transformers import pipeline
from duckduckgo_search import DDGS

# Load the logic model
pipe = pipeline("text-classification", model="roberta-large-mnli")

def fact_checker(claim, history):
    if not claim.strip():
        return "Please enter a claim.", history
    
    # 1. Search the web
    with DDGS() as ddgs:
        search_results = [r['body'] for r in ddgs.text(claim, max_results=2)]
    
    evidence = " ".join(search_results) if search_results else "No info found."
    
    # 2. AI Logic Check
    input_text = f"Claim: {claim} Evidence: {evidence}"
    result = pipe(input_text)
    label = result[0]['label']
    
    # 3. Format Result
    verdict = "✅ TRUE" if label == "ENTAILMENT" else "❌ FALSE" if label == "CONTRADICTION" else "🤔 UNCERTAIN"
    
    # 4. Update History Table
    new_row = {"Claim": claim, "Verdict": verdict}
    history.append(new_row)
    
    # Convert history list to a DataFrame for display
    df_history = pd.DataFrame(history)
    
    output_text = f"{verdict}\n\nTop Evidence: {evidence[:200]}..."
    return output_text, history, df_history

# --- The UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 AI Fact Checker with History")
    
    # This stores the history behind the scenes
    session_history = gr.State([])

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Enter Claim", placeholder="e.g., Earth is flat")
            check_btn = gr.Button("Check Fact", variant="primary")
        
        with gr.Column():
            output_verdict = gr.Textbox(label="AI Analysis")

    gr.Markdown("### 🕒 Recent Checks")
    history_table = gr.Dataframe(
        headers=["Claim", "Verdict"],
        datatype=["str", "str"],
        interactive=False
    )

    # Connect the button to the function
    check_btn.click(
        fn=fact_checker, 
        inputs=[input_text, session_history], 
        outputs=[output_verdict, session_history, history_table]
    )

if __name__ == "__main__":
    demo.launch()
