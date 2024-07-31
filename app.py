from transformers import pipeline
import gradio as gr

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, placeholder="Enter text here..."),
    outputs="text",
    title="Text Summarizer",
    description="Enter text to get a summarized version. This application uses the BART model for summarization."
)

iface.launch()