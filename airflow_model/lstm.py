from train import *
import csv
import os
import shutil
import matplotlib.pyplot as plt


def run_lstm():

    param_lstm = {
        'num_epochs': 1200,
        'batch_num': 1,
        'learning_rate': 0.008,
        'dropout': 0,
        'input_size': 5,
        'hidden_size': 7,
        'num_layers': 1,
        'num_classes': 24,  # should be same as pred_length
        'seq_length': 48,
        'pred_length': 24,
        'train_size': 55000,
        'validation_size': 5000,
        'test_size': 10000,
    }

    sc = MinMaxScaler()
    df_input = load_input()
    training_data = sc.fit_transform(df_input)
    x, y = sliding_windows(training_data, param_lstm['seq_length'], param_lstm['pred_length'], sc)
    dataX, dataY, trainX, trainY, validationX, validationY, testX, testY = train_test_split(x, y, param_lstm)

    fields = ['index'] + list(param_lstm.keys()) + ['optimizer', 'validation loss', 'test_mae_loss', 'test_mape_loss']

    dirname = os.path.dirname(__file__)
    results_path = os.path.join(dirname, 'results/model2v3_results_lstm')

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    os.chdir(results_path)
    for f in os.listdir():
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)

    results = []
    for criteria in ['huber_0.005', 'huber_0.01', 'huber_0.015', 'huber_0.02', 'huber_0.025', 'huber_0.03']:
        os.chdir(results_path)
        min_loss, index = -1, -1
        predict_df, true_df, model_to_save = None, None, None
        rows = []
        param_lstm['criterion'] = criteria

        count = 0
        for sq in [48]:
            param_lstm['seq_length'] = sq
            for hidden_size in [4]:
                param_lstm['hidden_size'] = hidden_size

                test_predict_df, test_true_df, test_predict_np, test_true_np, validation_loss, test_mae_loss, test_mape_loss, model = execute_and_save(
                    param_lstm, 'lstm', trainX, trainY, validationX, validationY, testX, testY)
                print('val_loss:', validation_loss)
                print('mae:', test_mae_loss)
                print('mape:', test_mape_loss)
                row = list(param_lstm.values())
                rows.append([count] + row + [validation_loss, test_mae_loss, test_mape_loss])
                if min_loss < 0 or min_loss > validation_loss:
                    min_loss = validation_loss
                    index = count
                    predict_df, true_df, model_to_save = test_predict_df, test_true_df, model
                count += 1

        filename = "{}_results_report.csv".format(criteria)
        # writing to csv file
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)
        os.mkdir(results_path + "{}_optimized_model".format(criteria))
        os.chdir(results_path + "{}_optimized_model".format(criteria))
        predict_df.to_csv('prediction_tesetY_results_{}(index:{}).csv'.format(criteria, index))
        true_df.to_csv('true_testY_results_{}(index:{}).csv'.format(criteria, index))
        torch.save(model_to_save, 'model_{}(index:{}).pth'.format(criteria, index))
        results.append(rows[index])

    os.chdir(results_path)
    filename = "report.csv"

    with open(filename, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(fields)
        csvwriter.writerows(results)


def demo_lstm():
    df_input = load_input()
    sc = MinMaxScaler()
    sc = sc.fit(df_input)
    data = pd.DataFrame(df_input, columns=df_input.columns)
    test = data.iloc[-10000:, :]
    sampleX = sc.transform(test[50 - 48:50]).reshape(1, 48, 5)
    sampleX = torch.tensor(sampleX).to('cuda')
    sampleX = sampleX.float()
    lstm_model = torch.load(
        'results/model2v3_results_lstm/huber_0.015_optimized_model/model_huber_0.015(index:0).pth')
    lstm_model.eval()
    lstm_model.to('cuda')

    pred = lstm_model(sampleX)
    min = sc.data_min_[1]
    max = sc.data_max_[1]

    pred = pred * (max - min) + min
    pred = pred.cpu()
    pred = pred.detach().numpy()
    plt.plot(test[50:50 + 24]['Primary Air Air Flow'], label='actual')
    plt.plot([i for i in range(63615 - 50, 63615 - 50 + 24)], pred[0], label='pred')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_lstm()

