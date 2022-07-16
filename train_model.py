import argparse
import os

import numpy as np
import pandas as pd
import torch

import model_utils
import preprocess
from datetime import datetime

torch.manual_seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--model', default='lstm', type=str, help='model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0, type=float, help='dropout')
    parser.add_argument('--bidirectional', default=False, type=bool, help='bidirectional')
    parser.add_argument('--num-layers', default=1, type=int, help='number of layers of LSTM')
    parser.add_argument('--embedding-size', default=256, type=int, help='embedding size')
    parser.add_argument('--hidden-size', default=128, type=int, help='hidden size')
    parser.add_argument('--vocab-size', default=20000, type=int, help='vocab size')
    parser.add_argument('--path', default=os.getcwd() + '/data/airline_sentiment_analysis.csv', type=str, help='path')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--early-stop', default=False, type=bool, help='early stop')
    parser.add_argument('--eval', default=False, type=bool, help='text')
    parser.add_argument('--save-model', default=False, type=bool, help='save model')

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, valid_dataset, text_field, label_field = preprocess.get_datasets(args.path, args.vocab_size)
    train_loader, valid_loader = preprocess.get_dataloaders(train_dataset, valid_dataset, args.batch_size, device)
    model = model_utils.get_model(args.model)(len(text_field.vocab), args.embedding_size, args.hidden_size,
                                              len(label_field.vocab), text_field, args.num_layers, args.dropout,
                                              args.bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    total_train_loss, total_train_accuracy, total_valid_loss, total_valid_accuracy = model.train_model(train_loader,
        valid_loader, optimizer, criterion, device, epochs=args.epochs, early_stop=args.early_stop)

    if args.eval:
        while True:
            text = input('Enter text: ')
            if text == 'exit':
                break
            prediction = model.pred(model, text, device, text_field)
            print(f'Prediction: {prediction}')
            print(f'Original text: {text}')
            print('\n')

    if args.save_model:
        path = args.path[:]
        del args.path
        time = str(datetime.now())
        save_dir_name = os.path.join(os.getcwd(), 'models', time)
        os.makedirs(save_dir_name, exist_ok=True)
        model.save(path=f'models/{time}')
        results = pd.DataFrame(columns=['train_loss', 'train_accuracy', 'valid_loss', 'valid_accuracy'])
        results['train_loss'] = total_train_loss
        results['train_accuracy'] = total_train_accuracy
        results['valid_loss'] = total_valid_loss
        results['valid_accuracy'] = total_valid_accuracy
        results.to_csv(f'{save_dir_name}/results.csv', index=False)
        with open(f'{save_dir_name}/config.txt', 'w') as f:
            f.write(str(args))
        model_utils.plot_result(total_train_loss, total_train_accuracy, total_valid_loss, total_valid_accuracy,
                                save_dir_name)
