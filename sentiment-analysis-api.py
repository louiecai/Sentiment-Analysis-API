import torch
from fastapi import FastAPI, Query

from model import RNN

app = FastAPI()

model_path = 'final_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN.load(model_path)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/eval")
def eval_text(text: str = Query('', description='The text to be evaluated.')) -> dict:
    return {'sentiment': model.predict(text, device)[0] if text != '' else None}
