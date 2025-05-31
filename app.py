import os

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define model path
model_dir = "./finetuned_model"

# Verify model directory exists
if not os.path.exists(model_dir):
    st.error(
        f"Model directory '{model_dir}' not found. Please ensure './finetuned_model' contains the fine-tuned model files."
    )
    st.stop()


# Load the model
@st.cache_resource
def load_model():
    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-math-7b-instruct",
            torch_dtype=torch.float16,  # Mixed precision
            device_map="auto",
            offload_buffers=True,
        )
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Initialize model and tokenizer
model_tokenizer = load_model()
if model_tokenizer is None:
    st.error(
        "Failed to initialize model. Check the error message above and ensure model files are correct."
    )
    st.stop()
model, tokenizer = model_tokenizer

# Streamlit app layout
st.title("Numerical Linear Algebra Q&A")
st.write(
    "Ask any numerical linear algebra question, and get a detailed answer powered by a fine-tuned model."
)

# Input question
question = st.text_area(
    "Enter your question:",
    placeholder="Example: Explain how to solve a system of linear equations using Gaussian elimination.",
    height=150,
)

# Generate and display answer
if st.button("Get Answer"):
    if question.strip():
        try:
            # Format prompt
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_length=1000, truncation=True, temperature=0.7, top_p=0.9
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.replace(prompt, "").strip()
            st.subheader("Answer:")
            st.markdown(answer, unsafe_allow_html=True)  # Render LaTeX
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
    else:
        st.warning("Please enter a question.")
