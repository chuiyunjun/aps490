import pandas as pd
import numpy as np
from tqdm.notebook import tqdm as tqdm
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import os
from models import *
random.seed(0)


def load_input():

    # dataset location
    dirname = os.path.dirname(__file__)
    file_name = os.path.join(dirname, 'datasets/VAV-1704-Airflow-analysis-v2.csv')
    df = pd.read_csv(file_name)

    # parameter selection
    date = 'DateTime_x'
    input1 = 'Temp (°C)'
    input2 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Primary Air.Air Flow'
    input3 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Zone Air.Temperature'
    input4 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Fin Tube Radiation.Valve Position'
    input5 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Primary Air.Air Flow Setpoint'

    input_list = [input1, input2, input3, input4, input5]
    df_input = df[input_list]
    df_input = df_input.dropna()

    return df_input


# Training setup
def sliding_windows(data, seq_length, pred_length, sc):
    x = []
    y = []
    samples = []
    for i in range(len(data) - seq_length - pred_length):
        sample = data[i:(i + seq_length + pred_length)]
        samples.append(sample)
    random.shuffle(samples)
    x = np.array(samples)[:, :seq_length, :]
    y = np.array(samples)[:, seq_length:, :]
    y = y[:, :, 1]
    return x, y


def train_test_split(x, y, param):
    train_size = param['train_size']
    validation_size = param['validation_size']
    test_size = param['test_size']

    dataX = torch.Tensor(np.array(x)).to('cuda')
    dataY = torch.Tensor(np.array(y)).to('cuda')

    trainX = torch.Tensor(np.array(x[0:train_size])).to('cuda')
    trainY = torch.Tensor(np.array(y[0:train_size])).to('cuda')

    validationX = torch.Tensor(np.array(x[train_size: train_size + validation_size])).to('cuda')
    validationY = torch.Tensor(np.array(y[train_size: train_size + validation_size])).to('cuda')

    # testY doesn't contain 0
    testX = []
    testY = []
    index = train_size + validation_size
    i = 0
    val_count = 0
    inval_count = 0
    while i < train_size and index < len(y):
        if np.all(y[index]):
            testX.append(x[index])
            testY.append(y[index])
            i += 1
            val_count += 1
        else:
            inval_count += 1
        index += 1
    print('val:', val_count, 'inval_count', inval_count)
    testX = torch.Tensor(np.array(testX)).to('cuda')
    testY = torch.Tensor(np.array(testY)).to('cuda')
    # testX = torch.Tensor(np.array(x[train_size+validation_size:train_size+validation_size+train_size])).to('cuda')
    # testY = torch.Tensor(np.array(y[train_size+validation_size:train_size+validation_size+train_size])).to('cuda')
    return dataX, dataY, trainX, trainY, validationX, validationY, testX, testY


def batch_split(x, y, param):
    l = len(x)
    if 'batch_size' in param:
        batch_size = param['batch_size']
        batch_num = l // param['batch_size'] if l // param['batch_size'] == 0 else l // param['batch_size'] + 1
    elif 'batch_num' in param:
        batch_num = param['batch_num']
        batch_size = l // param['batch_num'] if l // param['batch_num'] == 0 else l // param['batch_num'] + 1

    x_list, y_list = [], []
    for i in range(batch_num):
        if (i + 1) * batch_size > l:
            x_list.append(x[i * batch_size:])
            y_list.append(y[i * batch_size:])
        else:
            x_list.append(x[i * batch_size: (i + 1) * batch_size])
            y_list.append(y[i * batch_size: (i + 1) * batch_size])
    return x_list, y_list, batch_num


def train_and_evaluate(param, validationX, validationY, trainX, trainY, model):
    train_loss_list = torch.tensor([]).to('cuda')
    validation_loss_list = torch.tensor([]).to('cuda')

    trainX, trainY, num_batches = batch_split(trainX, trainY, param)
    validationX, validationY, num_batches = batch_split(validationX, validationY, param)

    optimizer = param["optimizer"]
    if param['criterion'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif param['criterion'] == 'mae':
        criterion = torch.nn.L1Loss()
    elif param['criterion'].startswith('huber'):
        delta = float(param['criterion'].split('_')[1])
        criterion = torch.nn.HuberLoss(delta=delta)
    elif param['criterion'] == 'nll':
        criterion = torch.nn.NLLLoss()
    else:
        print('Invalid criterion name: {}'.format(param['criterion']))

    # Train the model
    for epoch in tqdm(range(param['num_epochs'])):
        curr_batch_train_loss_list = torch.tensor([]).to('cuda')
        curr_batch_validation_loss_list = torch.tensor([]).to('cuda')
        for b in range(num_batches):
            outputs = model(trainX[b])
            optimizer.zero_grad()

            # obtain the loss function
            train_loss = criterion(outputs, trainY[b])

            curr_batch_train_loss_list = torch.cat(
                (curr_batch_train_loss_list, torch.tensor([train_loss.item()]).to('cuda')))

            train_loss.backward()

            optimizer.step()

            with torch.no_grad():
                pred_validationY = model(validationX[b])
                validation_loss = criterion(pred_validationY, validationY[b])
                curr_batch_validation_loss_list = torch.cat(
                    (curr_batch_validation_loss_list, torch.tensor([validation_loss.item()]).to('cuda')))

        train_loss_list = torch.cat(
            (train_loss_list, torch.tensor([curr_batch_train_loss_list.mean().item()]).to('cuda')))
        validation_loss_list = torch.cat(
            (validation_loss_list, torch.tensor([curr_batch_validation_loss_list.mean().item()]).to('cuda')))

        if (epoch + 1) % 50 == 0:
            print("Epoch: %d, train_loss: %1.5f, val_loss: %1.5f" % (
            epoch, curr_batch_train_loss_list.mean().item(), curr_batch_validation_loss_list.mean().item()))
    return model, train_loss_list, validation_loss_list


def execute_and_save(param, model_name, trainX, trainY, validationX, validationY, testX, testY):
    if param['criterion'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif param['criterion'] == 'mae':
        criterion = torch.nn.L1Loss()
    elif param['criterion'].startswith('huber'):
        delta = float(param['criterion'].split('_')[1])
        criterion = torch.nn.HuberLoss(delta=delta)
    elif param['criterion'] == 'nll':
        criterion = torch.nn.NLLLoss()
    else:
        print('Invalid criterion name: {}'.format(param['criterion']))

    if model_name == 'lstm':
        model = LSTM(param['num_classes'], param['input_size'], param['hidden_size'], param['num_layers'])
    elif model_name == 'gru':
        model = GRUNet(param['num_classes'], param['input_size'], param['hidden_size'], param['num_layers'])
    else:
        print('Invalid model name. Please double-check.')
        return

    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])
    param["optimizer"] = optimizer

    model, train_loss, validation_loss = train_and_evaluate(param, validationX, validationY, trainX, trainY, model)

    with torch.no_grad():
        test_predict = model(testX)
    test_data_predict = test_predict.data.cpu().numpy()
    test_data_true = torch.squeeze(testY).data.data.cpu().numpy()

    sc = MinMaxScaler()
    min = sc.data_min_[1]
    max = sc.data_max_[1]

    test_data_predict_np = test_data_predict * (max - min) + min
    test_data_predict_df = pd.DataFrame(test_data_predict_np)
    test_data_true_np = test_data_true * (max - min) + min
    test_data_true_df = pd.DataFrame(test_data_true_np)

    test_mae_loss = mean_absolute_error(test_data_true_np, test_data_predict_np)
    test_mape_loss = mean_absolute_percentage_error(test_data_true_np, test_data_predict_np)

    return test_data_predict_df, test_data_true_df, test_data_predict_np, test_data_true_np, validation_loss[
        -1].item(), test_mae_loss, test_mape_loss, model

