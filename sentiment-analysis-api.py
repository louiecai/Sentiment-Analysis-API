import torch
from fastapi import FastAPI

from model_utils import Namespace
from model import RNN

app = FastAPI()

model_path = 'models/2022-07-17 15:35:54.392705'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN.load(model_path)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/eval")
def eval_text(text: str = '') -> dict:
    return {'sentiment': model.predict(text, device)[0] if text != '' else None}
