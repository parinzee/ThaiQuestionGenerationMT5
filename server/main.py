import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
from enum import Enum


class Models(Enum):
    default = "mT5-small-thai-multiple-e2e-qg"
    separated = "mT5-small-thai-multiple-e2e-qg-sep"
    number_separated = "mT5-small-thai-multiple-e2e-qg-numsep"


class Args(BaseModel):
    model: Models = Models.default
    num_beams: int = 3
    max_length: int = 1024
    repetition_penalty: float = 2.5
    length_penalty: float = 1.0
    early_stopping: bool = True
    top_p: int = 50
    top_k: int = 20
    num_return_sequences: int = 1

app = FastAPI()

@app.post("/")
def root(input_text: str, args: Args):
    model = MT5ForConditionalGeneration.from_pretrained(f"parinzee/{args.model.value}", return_dict=True)
    tokenizer = MT5TokenizerFast.from_pretrained(f"parinzee/{args.model.value}")

    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True)
        generated_ids = model.generate(
            input_ids=input_ids,
            num_beams=args.num_beams,
            max_length=args.max_length,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
        )

        preds = [
            tokenizer.decode(
                g,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ) for g in generated_ids
        ]

    return preds
