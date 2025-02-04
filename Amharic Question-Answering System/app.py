import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the pre-trained Amharic QA model
model = AutoModelForQuestionAnswering.from_pretrained("./amharic_qa_model")
tokenizer = AutoTokenizer.from_pretrained("./amharic_qa_model")

def predict_answer(context, question):
    # Tokenize the inputs
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the most likely start and end positions of the answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Decode the answer from token IDs to a string
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    ).strip()

    # Handle cases where the model doesn’t return a valid answer
    if not answer or answer == tokenizer.pad_token or "[UNK]" in answer:
        return "Unable to find an appropriate answer."

    return answer

# Define Gradio interface
interface = gr.Interface(
    fn=predict_answer,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter the context (ማብራሪያ)...", label="Context (ማብራሪያ)"),
        gr.Textbox(lines=1, placeholder="Enter the question (ጥያቄ)...", label="Question (ጥያቄ)")
    ],
    outputs=gr.Textbox(label="Answer (መልስ)"),
    title="Amharic Question-Answering System",
    description="Provide a context and a question in Amharic to get the answer."
)

if __name__ == "__main__":
    interface.launch()
