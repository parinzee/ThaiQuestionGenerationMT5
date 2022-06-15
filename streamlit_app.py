import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("parinzee/mT5-small-thai-multiple-e2e-qg")
model = AutoModelForSeq2SeqLM.from_pretrained("parinzee/mT5-small-thai-multiple-e2e-qg")

st.title("mT5-small-thai-multiple-e2e-qg")
text_input = st.text_input("Command:")

if text_input:
    input_ids = tokenizer.encode(
        text_input, return_tensors="pt", add_special_tokens=True
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        num_beams=3,
        max_length=10000,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        top_p=50,
        top_k=20,
        num_return_sequences=1,
    )

    preds = [
        tokenizer.decode(
            g,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        for g in generated_ids
    ]

    st.write(preds[0])
