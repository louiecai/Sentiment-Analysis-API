import argparse
import os
from datetime import datetime

import pandas as pd
import torch
from model import RNN

import model_utils
import preprocess

if __name__ == '__main__':
    # CLI Arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--model-type', default='lstm', type=str, help='model type: lstm or gru')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0, type=float, help='dropout')
    parser.add_argument('--bidirectional', default=False, type=bool, help='bidirectional')
    parser.add_argument('--num-layers', default=1, type=int, help='number of layers of LSTM')
    parser.add_argument('--embedding-size', default=256, type=int, help='embedding size')
    parser.add_argument('--hidden-size', default=128, type=int, help='hidden size')
    parser.add_argument('--vocab-size', default=20000, type=int, help='vocab size')
    parser.add_argument('--path', default=os.path.join(os.getcwd(), 'data', 'airline_sentiment_analysis.csv'), type=str,
                        help='path')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--early-stop', default=False, type=bool, help='early stop')
    parser.add_argument('--eval', default=False, type=bool, help='enter evaluation mode')
    parser.add_argument('--save-model', default=True, type=bool, help='save model')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--damping', default=0.0, type=float, help='damping')
    parser.add_argument('--nesterov', default=False, type=bool, help='nesterov')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer: adam or sgd')

    args = parser.parse_args()
    print(args)

    # get device and preprocess data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, valid_dataset, text_field, label_field = preprocess.get_datasets(args.path, args.vocab_size)
    train_loader, valid_loader = preprocess.get_dataloaders(train_dataset, valid_dataset, args.batch_size, device)

    # get model, optimizer, criterion
    model = RNN(len(text_field.vocab), args.embedding_size, args.hidden_size, len(label_field.vocab), text_field,
                label_field, args.num_layers, args.dropout, args.bidirectional, args.model_type.lower() == 'lstm').to(
        device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay) if args.optimizer == 'adam' else torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, dampening=args.damping, nesterov=args.nesterov)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # call the train method on the model
    total_train_loss, total_train_accuracy, total_valid_loss, total_valid_accuracy = model.train_model(train_loader,
                                                                                                       valid_loader,
                                                                                                       optimizer,
                                                                                                       criterion,
                                                                                                       device,
                                                                                                       epochs=args.epochs,
                                                                                                       early_stop=args.early_stop)

    # evaluate the model with custom input
    if args.eval:
        while True:
            text = input('Enter text: ')
            if text == 'exit':
                break
            prediction = model.pred(model, text, device, text_field)
            print(f'Prediction: {prediction}')
            print(f'Original text: {text}')
            print('\n')

    # save the model
    if args.save_model:
        time = str(datetime.now())
        save_dir_name = os.path.join(os.getcwd(), 'models', time)
        # make directory if it doesn't exist
        os.makedirs(save_dir_name, exist_ok=True)

        # save the model
        model.save(path=f'models/{time}')

        # save the results as csv
        results = pd.DataFrame(columns=['train_loss', 'train_accuracy', 'valid_loss', 'valid_accuracy'])
        results['train_loss'] = total_train_loss
        results['train_accuracy'] = total_train_accuracy
        results['valid_loss'] = total_valid_loss
        results['valid_accuracy'] = total_valid_accuracy
        results.to_csv(f'{save_dir_name}/results.csv', index=False)

        # save the config as txt
        with open(f'{save_dir_name}/config.txt', 'w') as f:
            f.write(str(args))

        # save the plot of the results
        model_utils.plot_result(total_train_loss, total_train_accuracy, total_valid_loss, total_valid_accuracy,
                                save_dir_name)
