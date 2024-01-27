import streamlit as st
import pickle 
import torch
with open("fine_tuned_model.pkl", "rb") as f:
    loaded_model, loaded_tokenizer = pickle.load(f)

from transformers import GPT2LMHeadModel, GPT2Tokenizer,GenerationConfig

def generate_response(question, model, tokenizer, max_length=100):
    # Tokenize the input question
    
    model.config.pad_token_id = tokenizer.eos_token_id
    input_ids = tokenizer.encode(question, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    

    # Generate response from the model
    generation_config = GenerationConfig(
    max_length= max_length,
    temperature=0.7,
    num_return_sequences=1
    )

    output = model.generate(input_ids,max_length=generation_config.max_length, temperature=generation_config.temperature)

    # Decode the generated response
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_response


st.title("Q/A-Model")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFE4B5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add animation using HTML/CSS
st.markdown(
    """
    <style>
    @keyframes fadeInOut {
        0% {
            opacity: 0;
        }
        50% {
            opacity: 1;
        }
        100% {
            opacity: 0;
        }
    }

    .question-answer-animation {
        animation: fadeInOut 3s ease-in-out infinite;
    }
    </style>
    """
    , unsafe_allow_html=True
)

# Display animated text related to question-answers
st.markdown('<p class="question-answer-animation">Get Answers to your Questions...</p>', unsafe_allow_html=True)


question = st.text_input('Enter the question here')


if question:
    answer =  generate_response(question, loaded_model, loaded_tokenizer, max_length=100)
    st.write(answer)
        
        

