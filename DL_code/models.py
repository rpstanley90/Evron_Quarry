# Import installed packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import logsumexp
from smelu import SmeLU

class Simple1DCNN(torch.nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=41, stride=1)
        self.maxp1 = torch.nn.MaxPool1d(kernel_size=2,stride=2)
        
        self.layer2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=41, stride=1, dilation=2)
        self.maxp2 = torch.nn.MaxPool1d(kernel_size=2,stride=2)
        
        self.layer3 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=41, stride=1, dilation=3)
        self.maxp3 = torch.nn.MaxPool1d(kernel_size=2,stride=2)
        
        self.flatten1 = torch.nn.Flatten()
        
        self.linear1 = torch.nn.Linear(1280+10, 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.linear3 = torch.nn.Linear(256, 1)
        
        self.embedding = nn.Embedding(10, 10)
        
        self.smelu = SmeLU(beta=0.1)


        
    def forward(self, x):
        
        x, sites = torch.split(x, x.shape[-1]-10, dim=2)
        
        x = self.layer1(x)
        x = self.smelu(x)
        x = self.maxp1(x)

        x = self.layer2(x)
        x = self.smelu(x)
        x = self.maxp2(x)
        
        x = self.layer3(x)
        x = self.smelu(x)
        x = self.maxp3(x)
        
        
        x=self.flatten1(x)
        
        #sites = self.embedding(sites)
        sites=self.flatten1(sites)

        x = torch.cat((x,sites),dim=1)


        x = self.smelu(self.linear1(x))
        x = self.smelu(self.linear2(x))
        out = self.linear3(x)

        return out

class Standard_MLP(nn.Module):
    def __init__(self, network_size, **kwargs):
        super(Standard_MLP, self).__init__()

        # Dropout related settings
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout_type = kwargs['dropout_type']
        else:
            self.dropout_rate = 0
            self.dropout_type = 'identity'

        # Nonlinear layer setting
        if 'activation_func' in kwargs:
            self.nonlinear_type = kwargs['activation_func']
        else:
            self.nonlinear_type = 'relu'

        # Hidden Layer(s)
        self.hidden_layers = nn.ModuleList()
        for k in range(len(network_size)-2):
            self.hidden_layers.append(
                nn.ModuleDict({
                    'linear': nn.Linear(int(network_size[k]), int(network_size[k+1])),
                    'dropout': create_dropout_layer(self.dropout_rate, self.dropout_type),
                    'nonlinear': create_nonlinearity_layer(self.nonlinear_type)})
            )

        self.output = nn.ModuleDict({'linear': nn.Linear(
            int(network_size[-2]), int(network_size[-1]))})

    def forward(self, X):

        # loop through hidden layer Module List
        for hidden in self.hidden_layers:
            X = hidden['dropout'](X)
            X = hidden['linear'](X)
            X = hidden['nonlinear'](X)

        # pass through linear output layer
        X = self.output['linear'](X)

        return X

class Split_MLP(nn.Module):
    def __init__(self, network_size, **kwargs):
        super(Split_MLP, self).__init__()

        # Dropout related settings
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout_type = kwargs['dropout_type']
        else:
            self.dropout_rate = 0
            self.dropout_type = 'identity'

        # Nonlinear layer setting
        if 'nonlinear_type' in kwargs:
            self.nonlinear_type = kwargs['nonlinear_type']
        else:
            self.nonlinear_type = 'relu'

        # Learn heteroscedastic variance or not
        if 'learn_hetero' in kwargs:
            self.learn_hetero = kwargs['learn_hetero']

        # Setup layers
        # Hidden Layer(s) for mean regression
        self.mean_hidden_layers = nn.ModuleList()
        for i in range(len(network_size)-2):
            self.mean_hidden_layers.append(
                nn.ModuleDict({
                    'linear': nn.Linear(int(network_size[i]), int(network_size[i+1])),
                    'dropout': create_dropout_layer(self.dropout_rate, self.dropout_type),
                    'nonlinear': create_nonlinearity_layer(self.nonlinear_type)})
            )

        # Hidden layer(s) for variance regression
        self.noise_hidden_layers = nn.ModuleList()
        for k in range(len(network_size)-2):
            self.noise_hidden_layers.append(
                nn.ModuleDict({
                    'linear': nn.Linear(int(network_size[k]), int(network_size[k+1])),
                    'dropout': create_dropout_layer(self.dropout_rate, self.dropout_type),
                    'nonlinear': create_nonlinearity_layer(self.nonlinear_type)})
            )

        # Hetero noise
        self.output_noise = nn.ModuleDict(
            {'linear': nn.Linear(int(network_size[-2]), int(network_size[-1])//2)})

        # Output
        self.output_mean = nn.ModuleDict({'linear': nn.Linear(
            int(network_size[-2]), int(network_size[-1])//2)})

    def forward(self, X, **kwargs):

        # Make copy of input for split model
        mean = X
        noise = X

        # Forward through hidden layers
        for hidden in self.mean_hidden_layers:
            mean = hidden['dropout'](mean)
            mean = hidden['linear'](mean)
            mean = hidden['nonlinear'](mean)

        for hidden in self.noise_hidden_layers:
            noise = hidden['dropout'](noise)
            noise = hidden['linear'](noise)
            noise = hidden['nonlinear'](noise)

        # Output layers
        noise = self.output_noise['linear'](noise)
        mean = self.output_mean['linear'](mean)

        # Concat results
        result = torch.cat((mean, noise), 1)

        return result

# High level class for building NN regressor. Selects model architecture, optimizer, loss function from configuration

def create_dropout_layer(dropout_rate, dropout_type='identity'):
    if dropout_type == 'bernoulli':
        dropout_layer = nn.Dropout(dropout_rate)
    else:
        # No dropout at all
        dropout_layer = nn.Identity()

    return dropout_layer


def create_nonlinearity_layer(nonlinear_type='relu'):
    if nonlinear_type == 'relu':
        return nn.ReLU()
    elif nonlinear_type == 'tanh':
        return nn.Tanh()
    elif nonlinear_type == 'sigmoid':
        return nn.Sigmoid()
    elif nonlinear_type =='smelu':
        return SmeLU(beta=0.1)


def create_nonlinearity_layer_functional(nonlinear_type='relu'):
    if nonlinear_type == 'relu':
        return F.relu
    elif nonlinear_type == 'tanh':
        return F.tanh
    elif nonlinear_type == 'sigmoid':
        return F.sigmoid
    elif nonlinear_type =='smelu':
        return SmeLU


def train_ensemble(config, X_train, y_train, X_val=None, y_val=None):

    # Make sure data is shaped properly
    X_train = X_train.reshape(-1, int(config['network_size'][0]))
    y_train = y_train.reshape(-1, int(config['num_labels']))

    # Regressing heteroscedastic aleatoric uncertainty?
    if (config['network_size'][-1] == 2*config['num_labels']):
        config['learn_hetero'] = True
    else:
        config['learn_hetero'] = False

    # Initialize list to store trained models
    torch.cuda.empty_cache()
    model_list = []

    # Train ensemble of neural networks
    num_models_to_train = int(
        config['num_bootstraps'] + config['num_bootstraps_extra'])
    for k in range(0, num_models_to_train):

        print(f"\nBootstrap {k+1} of {num_models_to_train}")

        # Clean up memory
        torch.cuda.empty_cache()

        # Create NN regressor object and train model
        regressor = NN_Regressor(config).fit(X_train, y_train, X_val, y_val)

        # Move trained model to cpu to clear up GPU memory and add to list
        regressor.model.to("cpu")

        # Move model back to CPU and append to list of models
        model_list.append(regressor.model.cpu())

        # Clean up memory
        del regressor
        torch.cuda.empty_cache()

    if X_val is None:
        model_list.sort(key=lambda model: model.min_training_loss)
        model_list = model_list[0:config['num_bootstraps']]
    else:
        model_list.sort(key=lambda model: model.min_validation_loss)
        model_list = model_list[0:config['num_bootstraps']]

    return model_list


def standard_predict(model, X, device='cpu', loss_metric='mse'):
    """Perform model prediction"""

    # Standard Predictions
    torch.cuda.empty_cache()

    # Move data to GPU if chosen
    X = torch.from_numpy(X).type(torch.FloatTensor).to(
        device if torch.cuda.is_available() else 'cpu')

    # Move model to GPU if chosen
    model.to(device if torch.cuda.is_available() else 'cpu')

    # Set to eval mode
    model.eval()

    # No need for computing gradient
    with torch.no_grad():

        # Perform prediction and move back to CPU
        outs_standard = model(X).detach().cpu()

        # Move model to CPU
        model.cpu()

        # Slice prediction
        Yt_hat_standard = outs_standard[:, 0].numpy()

        if loss_metric == 'cfd_hau':
            logvar_standard = outs_standard[:, 1].numpy()
        else:
            logvar_standard = None

        # Clean up memory
        del X, outs_standard, model
        torch.cuda.empty_cache()

    return Yt_hat_standard, logvar_standard


def ensemble_predict(model_list, X, device='cpu', loss_metric='mse'):

    y_hat_list = []
    alea_logvar_list = []

    for k in range(0, len(model_list)):

        # Make prediction
        y_hat, alea_logvar = standard_predict(
            model_list[k], X, device=device, loss_metric=loss_metric)

        # Put results in a list
        y_hat_list.append(y_hat.reshape(1, -1))

        if alea_logvar is not None:
            alea_logvar_list.append(alea_logvar.reshape(1, -1))
        else:
            alea_logvar_list = None

    return y_hat_list, alea_logvar_list


def mc_predict(model, X, device='cpu', loss_metric='mse', T=10000):

    # Empty cuda cache to avoid memory leaks
    torch.cuda.empty_cache()

    y_hat = []
    logvar = []

    X = torch.from_numpy(X).type(torch.FloatTensor).to(
        device if torch.cuda.is_available() else 'cpu')

    model.to(device) if torch.cuda.is_available() else 'cpu'

    # MC Predictions
    model = model.train()
    for _ in range(T):

        outs_mc_predict = model(X)

        y_hat.append(outs_mc_predict[:, 0].cpu().data.numpy())

        if loss_metric == 'cfd_hau':
            logvar.append(outs_mc_predict[:, 1].cpu().data.numpy())
        else:
            logvar = None

    # Move model back to CPU to avoid GPU memory leaks
    model.eval()
    model.cpu()

    # Stack up results
    y_hat = np.vstack(y_hat).reshape(T, X.shape[0])
    if loss_metric == 'cfd_hau':
        logvar = np.vstack(logvar).reshape(T, X.shape[0])

    # Clean up memory
    torch.cuda.empty_cache()

    return y_hat, logvar


def gaussian_loglikelihood(y_hat_stacked, y_truth, T, tau=None, logvar_stacked=None):
    """Function for calculating the Gaussian log-likelihood across T models.
    This function as options for handling homoscedastic and heteroscedastic noise"""

    # Make sure our input data is shaped properly for logsumpexp to apply along correct axis
    assert y_hat_stacked.shape[0] == T
    assert y_truth.shape[1] == y_hat_stacked.shape[1]

    # Compute the logsumxp over the candidate models
    # Has homoscedastic and heteroscedastic options
    if logvar_stacked is None:
        ll = np.mean(logsumexp(-0.5 * tau * (y_truth - y_hat_stacked)
                     ** 2., 0) - np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
    else:
        assert logvar_stacked.shape[0] == T
        ll = np.mean(np.log(np.sum(1/np.sqrt(np.exp(logvar_stacked)) * np.exp(-0.5 * 1/(np.exp(
            logvar_stacked)) * (y_truth - y_hat_stacked)**2), 0)) - np.log(T) - 0.5*np.log(2*np.pi))

    return ll


def find_best_tau(y_truth, y_hat_arr, T, min_tau=0.1, max_tau=1000, num_taus=1000):

    best_tau = 0
    best_ll = -float('inf')
    tau_space = np.linspace(min_tau, max_tau, num_taus)
    ll_list = []
    for tau in tau_space:
        ll = gaussian_loglikelihood(y_hat_arr, y_truth, tau=tau, T=T)
        ll_list.append((tau, ll))
        if ll > best_ll:
            best_tau = tau
            best_ll = ll

    return best_tau, best_ll, np.asarray(ll_list)


def heteroscedastic_mse(yhat, y, yhat_logvar):
    losses = 1/2*(((yhat-y)**2)*torch.exp(-yhat_logvar) + yhat_logvar)
    loss = torch.mean(losses)
    return loss
