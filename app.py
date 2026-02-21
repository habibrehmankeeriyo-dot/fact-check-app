import gradio as gr
from transformers import pipeline
from duckduckgo_search import DDGS

# 1. Load the AI logic model
pipe = pipeline("text-classification", model="roberta-large-mnli")

def fact_checker(claim):
    if not claim.strip():
        return "Please enter a claim."

    # 2. Search the web for evidence
    with DDGS() as ddgs:
        search_results = [r['body'] for r in ddgs.text(claim, max_results=3)]
    
    if not search_results:
        return "Couldn't find any news about this on the web."

    # Join the top search results to use as 'evidence'
    evidence = " ".join(search_results)

    # 3. Compare Claim vs. Evidence
    # The model checks if the evidence supports or contradicts the claim
    input_text = f"Claim: {claim} Evidence: {evidence}"
    result = pipe(input_text)
    
    label = result[0]['label']
    score = result[0]['score']

    # Map the model labels to human-friendly results
    if label == "ENTAILMENT":
        return f"✅ LIKELY TRUE\n\nEvidence found: {search_results[0][:200]}..."
    elif label == "CONTRADICTION":
        return f"❌ LIKELY FALSE\n\nAI found conflicting info: {search_results[0][:200]}..."
    else:
        return f"🤔 UNCERTAIN\n\nI found some info, but it's not clear. Confidence: {score:.2%}"

# 4. The UI
demo = gr.Interface(
    fn=fact_checker,
    inputs=gr.Textbox(label="Enter a Statement or URL", placeholder="e.g., The moon is made of cheese"),
    outputs="text",
    title="Real-Time Fact Checker",
    description="This app searches the live web and uses AI to verify your claim."
)

if __name__ == "__main__":
    demo.launch()
