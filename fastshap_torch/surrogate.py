import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, BatchSampler
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm


class SoftCrossEntropyLoss(nn.Module):
    '''
    Soft cross entropy loss. Expects logits for pred, probabilities for target.
    '''

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        '''
        Evaluate loss.

        Args:
          pred:
          target:
        '''
        return - torch.mean(
            torch.sum(pred.log_softmax(dim=1) * target, dim=1))


class UniformSampler:
    '''
    For sampling player subsets with cardinality chosen uniformly at random.

    Args:
      num_players:
    '''

    def __init__(self, num_players):
        self.num_players = num_players

    def sample(self, batch_size):
        '''
        Generate sample.

        Args:
          batch_size:
        '''
        S = torch.ones(batch_size, self.num_players, dtype=torch.float32)
        num_included = (torch.rand(batch_size) * (self.num_players + 1)).int()
        # TODO ideally avoid for loops
        # TODO ideally pass buffer to assign samples in place
        for i in range(batch_size):
            S[i, num_included[i]:] = 0
            S[i] = S[i, torch.randperm(self.num_players)]

        return S


class Surrogate:
    '''
    Wrapper around surrogate model.

    Args:
      surrogate:
      num_features:
      groups:
    '''

    def __init__(self, surrogate, num_features, groups=None):
        # Store surrogate model.
        self.surrogate = surrogate

        # Store feature groups.
        if groups is None:
            self.num_players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.num_players = len(groups)
            device = next(surrogate.parameters()).device
            self.groups_matrix = torch.zeros(
                len(groups), num_features, dtype=torch.float32, device=device)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = 1

    def validate(self, loss_fn, data_loader):
        '''
        Calculate mean validation loss.

        Args:
          loss_fn:
          data_loader:
        '''
        with torch.no_grad():
            # Setup.
            device = next(self.surrogate.parameters()).device
            mean_loss = 0
            N = 0

            for x, y, S in data_loader:
                x = x.to(device)
                y = y.to(device)
                S = S.to(device)
                pred = self.__call__(x, S)
                loss = loss_fn(pred, y)
                N += len(x)
                mean_loss += len(x) * (loss - mean_loss) / N

        return mean_loss

    def train(self,
              train_data,
              val_data,
              batch_size,
              max_epochs,
              loss_fn,
              validation_samples,
              validation_batch_size,
              lr=1e-3,
              lookback=5,
              training_seed=None,
              validation_seed=None,
              bar=False,
              verbose=False):
        '''
        Train surrogate model.

        Args:
          train_data:
          val_data:
          batch_size:
          max_epochs:
          loss_fn:
          validation_samples:
          validation_batch_size:
          lr:
          lookback:
          training_seed:
          validation_seed:
          verbose:
        '''
        # Unpack and convert data.
        x_train, y_train = train_data
        x_val, y_val = val_data
        if isinstance(x_train, np.ndarray):
            x_train = torch.tensor(x_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            x_val = torch.tensor(x_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
        elif isinstance(x_train, torch.Tensor):
            pass
        else:
            raise ValueError('data must be torch.Tensor or np.ndarray')

        # Set up train data loader.
        train_set = TensorDataset(x_train, y_train)
        random_sampler = RandomSampler(
            train_set, replacement=True,
            num_samples=int(np.ceil(len(x_train) / batch_size))*batch_size)
        batch_sampler = BatchSampler(
            random_sampler, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler)

        # Prepare validation dataset.
        sampler = UniformSampler(self.num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        S_val = sampler.sample(len(x_val) * validation_samples)
        x_val_repeat = x_val.repeat(validation_samples, 1)
        y_val_repeat = y_val.repeat(validation_samples, 1)
        val_set = TensorDataset(
            x_val_repeat, y_val_repeat, S_val)
        val_loader = DataLoader(
            val_set, batch_size=validation_batch_size)

        # Setup for training.
        surrogate = self.surrogate
        device = next(surrogate.parameters()).device
        optimizer = optim.Adam(surrogate.parameters(), lr=lr)
        best_loss = self.validate(loss_fn, val_loader).item()
        best_epoch = 0
        best_model = deepcopy(surrogate)
        loss_list = [best_loss]
        if training_seed is not None:
            torch.manual_seed(training_seed)

        # Epoch iterable.
        if bar:
            epoch_iter = tqdm(range(max_epochs), desc='Epochs')
        else:
            epoch_iter = range(max_epochs)

        for epoch in epoch_iter:
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Batches', leave=False,
                                  total=len(train_loader))
            else:
                batch_iter = train_loader

            for x, y in batch_iter:
                # Prepare data.
                x = x.to(device)
                y = y.to(device)

                # Generate subsets.
                S = sampler.sample(batch_size).to(device=device)

                # Make predictions.
                pred = self.__call__(x, S)
                loss = loss_fn(pred, y)

                # Optimizer step.
                loss.backward()
                optimizer.step()
                surrogate.zero_grad()

            # Print progress.
            val_loss = self.validate(loss_fn, val_loader).item()
            loss_list.append(val_loss)
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Val loss = {:.4f}'.format(val_loss))
                print('')

            # Check if best model.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(surrogate)
                best_epoch = epoch
                if verbose:
                    print('New best epoch, loss = {:.4f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early')
                break

        # Clean up.
        for param, best_param in zip(surrogate.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        self.loss_list = loss_list

    def __call__(self, x, S):
        '''
        Evaluate surrogate model.

        Args:
          x:
          S:
        '''
        if self.groups_matrix is not None:
            S = torch.mm(S, self.groups_matrix)

        return self.surrogate((x, S))
