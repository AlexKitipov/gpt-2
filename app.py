from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./")
model = GPT2LMHeadModel.from_pretrained("./")
model.eval()


class Prompt(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(data: Prompt):
    inputs = tokenizer.encode(data.prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=80, temperature=0.9, top_p=0.95)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}
