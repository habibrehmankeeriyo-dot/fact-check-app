import gradio as gr
import os
import pandas as pd
from groq import Groq
from duckduckgo_search import DDGS
from newspaper import Article

# Initialize Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def process_input(user_input):
    """Detects if input is a URL or Text and extracts content."""
    if user_input.startswith(("http://", "https://")):
        try:
            article = Article(user_input)
            article.download()
            article.parse()
            return f"NEWS ARTICLE: {article.title}\n\n{article.text[:1500]}"
        except:
            return "Error: Could not read this URL. Please check the link."
    return user_input

def advanced_audit(claim_or_url, history):
    # 1. Pre-process Input
    content = process_input(claim_or_url)
    if "Error" in content: return content, [], pd.DataFrame(history), 0
    
    # 2. Search for Live Context
    sources = []
    with DDGS() as ddgs:
        # We search the first 100 characters of the content to find related news
        search_query = claim_or_url if len(claim_or_url) < 100 else content[:100]
        results = list(ddgs.text(search_query, max_results=4))
        search_text = "\n".join([f"Source: {r['title']} - {r['body']}" for r in results])
        sources = [[r['title'], r['href']] for r in results]

    # 3. Groq Reasoning (Llama 3.3 70B)
    prompt = f"""
    SYSTEM: You are an elite AI fact-checker. 
    TASK: Analyze the 'Content' against the 'Web Evidence'.
    FORMAT:
    ### VERDICT: [TRUE | FALSE | MISLEADING | UNCERTAIN]
    ### CONFIDENCE: [0-100]%
    ### SUMMARY: [Max 2 sentences]
    ### ANALYSIS: [Bullet points of key facts]

    Content to Check: {content}
    Web Evidence: {search_text}
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.1
    ).choices[0].message.content

    # 4. UI Updates
    # Logic to extract the percentage for the gauge
    try:
        conf_score = int(response.split("CONFIDENCE:")[1].split("%")[0].strip())
    except:
        conf_score = 50

    history.append({"Input": claim_or_url[:30]+"...", "Score": f"{conf_score}%"})
    
    return response, sources, pd.DataFrame(history), conf_score

# --- Advanced Dashboard UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🛡️ News Audit & Fact-Check Dashboard")
    state_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            input_data = gr.Textbox(
                label="Paste News URL or Text Claim", 
                placeholder="https://bbc.com/news... OR 'The moon is made of cheese'",
                lines=3
            )
            run_btn = gr.Button("🚀 Start Deep Audit", variant="primary")
        
        with gr.Column(scale=1):
            # Visual Gauge for Confidence
            gauge = gr.Slider(0, 100, label="AI Confidence Level", interactive=False)
            status_light = gr.Markdown("🟢 **System Online** | API Connected")

    with gr.Tabs():
        with gr.Tab("📋 Audit Report"):
            report_out = gr.Markdown("Waiting for input...")
        
        with gr.Tab("🔗 Verified Sources"):
            source_out = gr.Dataframe(headers=["Title", "Source URL"], interactive=False)
            
        with gr.Tab("📜 Session History"):
            log_out = gr.Dataframe(headers=["Input", "Score"])

    run_btn.click(
        fn=advanced_audit,
        inputs=[input_data, state_history],
        outputs=[report_out, source_out, log_out, gauge]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
