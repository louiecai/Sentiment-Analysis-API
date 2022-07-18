# Sentiment-Analysis-API

By Louie Cai.

## Environment Setup

Clone this repository:
```bash
git clone https://github.com/louie-cai/Sentiment-Analysis-API.git
cd Sentiment-Analysis-API
```

Create a new conda environment (Python version: `3.9.12`) and install the necessary packages:

```bash
pip install -r requirements.txt
```

## Training the model

`python train_model.py -h` prints out a list of options to tune the hyperparameters.
It has the option to save the model, the results of the training, a plot of the results, and the configuration of the
model. With the `--eval` flag, it creates an interactive terminal session to test the model.

There are two model types: LSTM and GRU. The script takes in a `.csv` file and load it into dataloaders for training.
The model is trained for a number of epochs (or with early stopping), and the relevant data is saved in a directory
named after the time it was trained.

## Running FastAPI

```bash
uvicorn sentiment-analysis-api:app --reload
```

To choose a model to run, set the `model_path` variable in `sentiment-analysis-api.py` to the path of the model (defaults to `final_model.pt`).
