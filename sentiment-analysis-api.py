import os

import numpy as np
import torch
from fastapi import FastAPI

import preprocess
from model_utils import get_model, load_model, model_predict, Namespace

app = FastAPI()

model_path = "Namespace(random_seed=42, batch_size=128, epochs=50, model='lstm', lr=0.001, dropout=0, bidirectional=False, num_layers=1, embedding_size=128, hidden_size=128, vocab_size=20000, weight_decay=0.0, early_stop=True, eval=False, save_model=True)"
args = eval(model_path)
path = open(os.path.join(os.getcwd(), 'models', model_path, 'path.txt'), 'r').readline()
torch.manual_seed(args.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.random_seed)
np.random.seed(args.random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_dataset, valid_dataset, text_field, label_field = preprocess.get_datasets(path, args.vocab_size)
# train_loader, valid_loader = preprocess.get_dataloaders(train_dataset, valid_dataset, args.batch_size, device)
# model = get_model(args.model)(len(text_field.vocab), args.embedding_size, args.hidden_size, len(label_field.vocab),
#                               args.num_layers, args.dropout, args.bidirectional).to(device)

model, text_field = load_model(os.path.join('models', model_path))


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/eval")
def eval_text(text: str = '') -> str:
    return str(model_predict(model, text, device, text_field)) if text != '' else 'Please enter text to evaluate'
