import torch
from prediction.model.gru import GRUNet

from prediction.model.lstm import LSTM


class ModelConfig:
    def __init__(self, params):
        self.num_classes = params.get('pred_length')
        self.epoch_num = params.get('epoch_num')
        self.learning_rate = params.get('learning_rate')
        self.hidden_size = params.get('hidden_size')
        self.num_layers = params.get('num_layers')
        self.input_size = params.get('input_size')
        self.parse_loss(params)
        self.model = params.get('model')
        
    def get_epoch_num(self):
        return self.epoch_num
    
    def get_learning_rate(self):
        return self.learning_rate

    def get_loss(self):
        return self.loss
    
    def parse_loss(self, params):
        if params.get('loss') == 'mse':
            self.loss = torch.nn.MSELoss()
        elif params.get('loss') == 'mae':
            self.loss = torch.nn.L1Loss()
        elif params.get('loss').startswith('huber'):
            delta = float(params.get('loss').split('_')[1])
            self.loss = torch.nn.HuberLoss(delta=delta)


class Prediction:
    def __init__(self, config, device):
        self.device = device
        self._config = config
        self.model = self.parse_model()
        
    def get_config(self):
        return self._config
    
    def get_model(self):
        return self.model
    
    def parse_model(self):
        if self.get_config().model == 'lstm':
            return LSTM(self.get_config().num_classes, self.get_config().input_size, self.get_config().hidden_size, self.get_config().num_layers, self.device)
        elif self.get_config().model == 'gru':
            return GRUNet(self.get_config().num_classes, self.get_config().input_size, self.get_config().hidden_size, self.get_config().num_layers, self.device)
        else:
            print('Input Error: invalid model argument')
            return
