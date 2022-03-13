# Predictive Building System Maintenance -- Anomaly Detection

## Training
You may use the following command to train the model.
```shell
$ python -m predection.main train
```
Keyword arguments:
- (optional)`data_root`: training dataset directory root. Default: `./prediction/datasets/train/`
- (optional)`output_root`: output directory root. Default: `./output/`
- (optional)`seed`: random seed. Default: `42`
- (optional)`seq_length`: squence length of the sliding window. Default: `48`
- (optional)`pred_length`: prediction length of the sliding window. Default: `24`
- (optional)`epoch_num`: number of epochs. Default: `1000`
- (optional)`learning_rate`: learning rate of the adam optimizer. Default: `0.0008`
- (optional)`hidden_size`: hidden_size of RNN . Default: `7`
- (optional)`num_layers`: num_layers of RNN. Default: `1`
- (optional)`loss`: training loss function. (`mae`, 'mse', or `huber_<delta>`) Default: `huber_0.022`
- (optional)`model`: RNN model type (`lstm` or `gru`). Default: `gru`

Example:
```shell
$ python -m prediction.main train --output_root=./output/
```

## Validate
You may use the following command to train the model.
```shell
$ python -m prediction.main validate --output_root=<output directory root> --checkpoint_path=<checkpoint path>
```
Keyword arguments:
- `checkpoint_path`: path to the model that will be evaulated
- `output_root`: the output directory root
- (optional)`data_root`: validation dataset directory root. Default: `./prediction/datasets/valid/`
- (optional)`seed`: random seed. Default: `42`
- (optional)`seq_length`: squence length of the sliding window. Default: `48`
- (optional)`pred_length`: prediction length of the sliding window. Default: `24`

Example:
```shell
$ python -m prediction.main validate --output_root=./output/ --checkpoint_path=./output/model.pth
```
