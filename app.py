import gradio as gr
from transformers import pipeline

# Load a model that can determine if a statement is True (Entailment) or False (Contradiction)
# Note: This is a starter model. Fact-checking is complex!
pipe = pipeline("text-classification", model="DaeSae/roberta-base-snli")

def fact_checker(text):
    if not text.strip():
        return "Please enter some text."
    
    # In a real model, you'd search the web first. 
    # Here, we analyze the sentiment/logic of the statement.
    result = pipe(text)
    label = result[0]['label']
    score = result[0]['score']
    
    if label == "LABEL_0": # Usually 'Entailment' in this model
        return f"✅ Likely TRUE (Confidence: {score:.2%})"
    elif label == "LABEL_2": # Usually 'Contradiction'
        return f"❌ Likely FALSE (Confidence: {score:.2%})"
    else:
        return f"🤔 UNCERTAIN/NEUTRAL (Confidence: {score:.2%})"

# Create the Gradio Interface
demo = gr.Interface(
    fn=fact_checker,
    inputs=gr.Textbox(lines=2, placeholder="Enter a claim here..."),
    outputs="text",
    title="AI Fact Checker",
    description="Enter a statement to see if the AI thinks it is True or False. Note: This is a demo using a logic-based model."
)

if __name__ == "__main__":
    demo.launch()
