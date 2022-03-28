import os
from pickletools import optimize
import random
from click import option

import numpy as np
import torch
import streamlit as st
from tqdm import tqdm
import csv
from prediction.model.config import ModelConfig, Prediction
from prediction.data import Data, format_path, sliding_windows
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# def mean_absolute_percentage_error(y_true, y_pred):
#   y_true, y_pred = np.array(y_true), np.array(y_pred)
#   return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100

def train(
    data_root: str = './prediction/datasets/train/',
    output_root: str = './output/',
    seed: int = 42,
    seq_length: int = 48,
    pred_length: int =24,
    epoch_num: int = 1000,
    learning_rate: float = 0.0008,
    hidden_size: int = 7,
    num_layers: int = 1,
    loss: str = 'huber_0.022',
    model: str = 'gru',
    option: str = 'V',
    ui: bool = True
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    data_root = format_path(data_root)
    output_root = format_path(output_root)

    data = Data(option=option, data_root=data_root)
    normalized_data = data.get_normalized_data()
    
    params = {
        'seq_length': seq_length,
        'pred_length': pred_length,
        'epoch_num': epoch_num,
        'learning_rate': learning_rate,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'input_size': normalized_data.shape[1],
        'loss': loss,
        'model': model
    }
    
    trainX, trainY = sliding_windows(normalized_data, seq_length, pred_length)

    trainX = trainX.to(device)
    trainY = trainY.to(device)

    model_config = ModelConfig(params)
    model = Prediction(model_config, device).get_model()
    model.to(device)
    
    loss = model_config.get_loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.get_learning_rate())
    
    train_loss_list = []
    for epoch in tqdm(range(model_config.get_epoch_num())):
        
        outputs = model(trainX)
        optimizer.zero_grad()

        # obtain the loss function
        train_loss = loss(outputs, trainY)
        train_loss.backward()
        optimizer.step()
        train_loss_list.append(train_loss)
        
        if (epoch + 1) % 50 == 0:
            print("Epoch: %d, train_loss: %1.5f" % (epoch, train_loss_list[-1].item()))
    
    if not os.path.exists(output_root):
      os.mkdir(output_root)

    if option =='V':
        torch.save(model, output_root + 'ValveModel.pth')
    else:
        torch.save(model, output_root + 'AirFlowModel.pth')
    return 0


def validate(
    checkpoint_path: str,
    output_root: str = './output/',
    data_root: str = './prediction/datasets/valid/',
    seed: int = 42,
    seq_length: int = 48,
    pred_length: int =24,
    option: str = 'V',
):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    data_root = format_path(data_root)
    output_root = format_path(output_root)
    
    data = Data(option=option, data_root=data_root)
    normalized_data = data.get_normalized_data()
    
    ValidX, ValidY = sliding_windows(normalized_data, seq_length, pred_length, shuffle=False)

    ValidX = ValidX.to(device)

    
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    
    with torch.no_grad():
        pred_validY = model(ValidX)
    
    pred_validY = pred_validY.cpu().numpy()
    ValidY = ValidY.cpu().numpy()
    pred_validY = data.recover_y(pred_validY)
    ValidY = data.recover_y(ValidY)
    

    mae = mean_absolute_error(pred_validY, ValidY)
    
    # mape = mean_absolute_percentage_error(pred_validY, ValidY)
    
    ValidX = data.recover_x(ValidX.cpu().numpy())
    
    visualize_sample(option, pred_validY, ValidY)

    if not os.path.exists(output_root):
        os.mkdir(output_root)
    
    fields = ['mae']
    values = [mae]
    filename = "report.csv"
    with open(output_root + filename, 'w') as csvfile: 
      csvwriter = csv.writer(csvfile) 
      csvwriter.writerow(fields) 
      csvwriter.writerow(values)
    return 0

def visualize_sample(option, pred_validY, ValidY):
    index = 190

    time_index_pred = np.arange(ValidY.shape[1])
    plt.plot(time_index_pred, ValidY[index,:], label="actual value")
    plt.plot(time_index_pred, pred_validY[index], label="pred value")
    plt.xlabel('time index')

    if option == 'V':
        plt.ylabel('Valve Position')
        plt.title('Valve Position Prediction in Next 2 hours')
    else:
        plt.ylabel('Primary Airflow ')
        plt.title(f'Primary Airflow Prediction in Next 2 hours')

    plt.savefig('sample.png')
    
    
    
    
    


if __name__ == '__main__':
    import fire
    fire.Fire()
