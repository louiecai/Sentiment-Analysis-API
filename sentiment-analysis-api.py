import os

import numpy as np
import torch
from fastapi import FastAPI

from model_utils import get_model, Namespace

app = FastAPI()

model_path = 'models/2022-07-16 15:27:27.014567'
args = eval(open(os.path.join(model_path, 'config.txt')).read())
torch.manual_seed(args.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.random_seed)
np.random.seed(args.random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(args.model).load(model_path)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/eval")
def eval_text(text: str = '') -> dict:
    return {'sentiment': model.predict(text, device)[0] if text != '' else 'Please enter text to evaluate'}
