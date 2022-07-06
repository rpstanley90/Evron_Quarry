import models
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from early_stopping import EarlyStopping

class Conductor:
    def __init__(self, config, **kwargs):

        # Set configuration as attribute for easy access
        self.config = config
        self.max_epoch = int(config['train_epochs'])

        # Select model architecture
        if self.config['model_architecture'] == 'split_mlp':
            self.model = models.Split_MLP(network_size=self.config['network_size'], dropout_rate=self.config['dropout'],
                                   dropout_type='bernoulli', activation_func=self.config['activation_func'])
        elif self.config['model_architecture'] == 'standard_mlp':
            self.model = models.Standard_MLP(network_size=self.config['network_size'], dropout_rate=self.config['dropout'],
                                      dropout_type='bernoulli', activation_func=self.config['activation_func'])
        elif self.config['model_architecture'] == '1dcnn':
            self.model=models.Simple1DCNN()


        self.model.to(self.config['device']
                      if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.AdamW(self.model.parameters(
        ), weight_decay=self.config['weight_decay'], lr=self.config['learning_rate'])

        self.model.min_training_loss = float("Inf")
        self.model.min_validation_loss = float("Inf")

        # Choose loss criterion and optimizer
        if config['loss_metric'] == 'cfd_hau':
            self.criterion = models.heteroscedastic_mse
        elif config['loss_metric'] == 'mse' or config['loss_metric'] == 'switch':
            self.criterion = nn.MSELoss()

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        # Create training dataset and data loader
        X = torch.from_numpy(X_train).type(torch.FloatTensor)
        y = torch.from_numpy(y_train).type(torch.FloatTensor)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], drop_last=False, shuffle=True)

        # Create validation dataset and data loader
        val_data_present = False
        if X_val is not None and y_val is not None:
            val_data_present = True
            X_val = torch.from_numpy(X_train).type(torch.FloatTensor)
            y_val = torch.from_numpy(y_train).type(torch.FloatTensor)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.config['batch_size'], drop_last=False, shuffle=True)

        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.config['scheduler_patience'], factor=0.75, verbose=self.config['verbose'])
        #scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        # Check early stopping criteria
        early_stopping = EarlyStopping(
            patience=self.config['patience'], delta=self.config['min_delta'], verbose=False)

        # Loop over training and validation data for number of epochs unless early stopping criteria met
        for epoch in range(self.max_epoch):
            train_losses_batch = []
            val_losses_batch = []

            # For each batch in training loader
            for X_batch, y_batch in train_loader:

                # Set model to train mode
                self.model.train()

                # Move data to GPU if selected, else stay on CPU
                X_batch = X_batch.to(
                    self.config['device'] if torch.cuda.is_available() else 'cpu')
                y_batch = y_batch.to(
                    self.config['device'] if torch.cuda.is_available() else 'cpu')

                # Zero gradient values
                self.optimizer.zero_grad()

                # Forward pass through model
                output = self.model(X_batch)

                # If learning heteroscedastic aleatoric uncertainty, split output
                if self.config['loss_metric'] == 'cfd_hau':
                    mean, log_var = output.split(
                        [int(self.config['num_labels']), int(self.config['num_labels'])], dim=1)
                else:
                    mean = output

                xyz = mean.detach().cpu().numpy()

                # Compute loss
                if self.config['loss_metric'] == 'cfd_hau':
                    loss = self.criterion(y_batch, mean, log_var)
                else:
                    loss = self.criterion(y_batch.T, mean.T)

                # Grab loss value
                train_loss = loss.detach().item()

                # Backpropagate loss
                loss.backward()

                # Perform optimization step
                self.optimizer.step()

                # Append training loss
                train_losses_batch.append(train_loss)

                # Clean up CUDA memory
                del loss, output

            # Compute mean training loss for epoch
            training_loss = np.mean(train_losses_batch)

            if (training_loss < self.model.min_training_loss):
                self.model.min_training_loss = training_loss

            # Check current model against validation dataset, if present
            if val_data_present:
                with torch.no_grad():

                    # Set to eval mode
                    self.model.eval()

                    # For each batch in validation loader
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch = X_val_batch.to(
                            self.config['device'] if torch.cuda.is_available() else 'cpu')
                        y_val_batch = y_val_batch.to(
                            self.config['device'] if torch.cuda.is_available() else 'cpu')

                        # Compute model output
                        output_val = self.model(X_val_batch)

                        # If learning heteroscedastic aleatoric uncertainty, split output
                        if self.config['loss_metric'] == 'cfd_hau':
                            mean_val, log_var_val = output_val.split(
                                [int(self.config['num_labels']), int(self.config['num_labels'])], dim=1)
                        else:
                            mean_val = output_val

                        # Compute loss
                        if self.config['loss_metric'] == 'cfd_hau':
                            loss = self.criterion(
                                y_val_batch, mean_val, log_var_val)
                        else:
                            loss = self.criterion(y_val_batch, mean_val)

                        # Grab loss value
                        val_loss = loss.detach().item()

                        # Append loss to list
                        val_losses_batch.append(val_loss)

                        # Clean up CUDA memory
                        del loss, output_val

                    if (val_loss < self.model.min_validation_loss):
                        self.model.min_validation_loss = val_loss

                    # Compute mean validation loss for epoch
                    validation_loss = np.mean(val_losses_batch)

            # Print training stats
            if self.config['verbose'] and epoch % self.config['log_thinning'] == 0:

                if val_data_present:
                    print(
                        f'Epoch {epoch+1} training loss: {training_loss:.7f}; validation loss {validation_loss:.7f}')
                else:
                    print(
                        f'Epoch {epoch+1} training loss: {training_loss:.7f}')

            # Adjust lr schedule if necessary
            scheduler.step(training_loss)

            # Switch loss function if set by configuration file
            if self.config['loss_metric'] == 'switch' and epoch == self.config['warmup_epochs']:
                print('Switching loss function')
                self.criterion = heteroscedastic_mse

            # Stop early if criteria met
            if self.config['early_stopping']:
                if val_data_present:
                    early_stopping(validation_loss, self.model)
                else:
                    early_stopping(training_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping criteria reached. Stopping training now.")
                break

        # Clean up memory
        del scheduler, self.optimizer
        torch.cuda.empty_cache()

        return self
