import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy


class ShapleySampler:
    '''
    For sampling player subsets from the Shapley distribution.

    Args:
      num_players:
    '''

    def __init__(self, num_players):
        arange = torch.arange(1, num_players)
        w = 1 / (arange * (num_players - arange))
        w = w / torch.sum(w)
        self.categorical = Categorical(probs=w)
        self.num_players = num_players
        self.tril = torch.tril(
            torch.ones(num_players - 1, num_players, dtype=torch.float32),
            diagonal=0)

    def sample(self, batch_size, paired_sampling):
        '''
        Generate sample.

        Args:
          batch_size:
          paired_sampling:
        '''
        num_included = 1 + self.categorical.sample([batch_size])
        S = self.tril[num_included - 1]
        # TODO ideally avoid for loops
        # TODO can't figure out how to shuffle rows independently
        # TODO don't want to generate paired samples in parallel and force even num_samples
        for i in range(batch_size):
            if paired_sampling and i % 2 == 1:
                S[i] = 1 - S[i - 1]
            else:
                S[i] = S[i, torch.randperm(self.num_players)]
        return S


def additive_efficient_normalization(pred, grand, null):
    '''
    Apply additive efficient normalization.

    Args:
      pred:
      grand:
      null:
    '''
    gap = (grand - null) - torch.sum(pred, dim=1)
    return pred + gap.unsqueeze(1) / pred.shape[1]


def calculate_grand_coalition(x, imputer, batch_size, link, device):
    '''
    Calculate the value of grand coalition for each x.

    Args:
      x:
      imputer:
      batch_size:
      num_players:
      link:
      device:
    '''
    ones = torch.ones(batch_size, imputer.num_players, dtype=torch.float32,
                      device=device)
    with torch.no_grad():
        grand = []
        for i in range(int(np.ceil(len(x) / (batch_size)))):
            start = i * batch_size
            end = min(start + batch_size, len(x))
            grand.append(link(
                imputer(x[start:end].to(device), ones[:(end-start)])))

        # Concatenate and return.
        grand = torch.cat(grand)
        if len(grand.shape) == 1:
            grand = grand.reshape(-1, 1)

    return grand


def validate(val_loader, imputer, explainer, null, link, normalization):
    '''
    Calculate mean validation loss.

    Args:
      val_loader:
      imputer:
      explainer:
      null:
      link:
      normalization:
    '''
    with torch.no_grad():
        # Setup.
        device = next(explainer.parameters()).device
        mean_loss = 0
        N = 0
        loss_fn = nn.MSELoss()

        for x, grand, S, values in val_loader:
            # Move to device.
            x = x.to(device)
            S = S.to(device)
            grand = grand.to(device)
            values = values.to(device)

            # Evaluate explainer.
            pred = explainer(x)
            pred = pred.reshape(len(x), imputer.num_players, -1)
            if normalization:
                pred = normalization(pred, grand, null)

            # Evaluate loss.
            approx = null + torch.matmul(S, pred)
            loss = loss_fn(approx, values)

            # Update average.
            N += len(x)
            mean_loss += (loss - mean_loss) / N

    return mean_loss


class FastSHAP:
    '''
    Wrapper around FastSHAP explanation model.

    Args:
      explainer:
      imputer:
      normalization:
      link:
    '''

    def __init__(self,
                 explainer,
                 imputer,
                 normalization='additive',
                 link=None):
        # Set up explainer, imputer and link function.
        self.explainer = explainer
        self.imputer = imputer
        self.num_players = imputer.num_players
        self.null = None
        if link is None:
            self.link = nn.Identity()
        elif isinstance(link, nn.Module):
            self.link = link
        else:
            raise ValueError('unsupported link function: {}'.format(link))

        # Set up normalization.
        if normalization is None:
            self.normalization = normalization
        elif normalization == 'additive':
            self.normalization = additive_efficient_normalization
        else:
            raise ValueError('unsupported normalization: {}'.format(
                normalization))

    def train(self,
              train_data,
              val_data,
              batch_size,
              num_samples,
              max_epochs,
              lr=2e-4,
              min_lr=1e-5,
              lr_factor=0.5,
              eff_lambda=0,
              paired_sampling=True,
              validation_samples=None,
              lookback=5,
              training_seed=None,
              validation_seed=None,
              verbose=False):
        '''
        Train explainer model.

        Args:
          train_data:
          val_data:
          batch_size:
          num_samples:
          max_epochs:
          lr:
          min_lr:
          lr_factor:
          eff_lambda:
          paired_sampling:
          validation_samples:
          lookback:
          training_seed:
          validation_seed:
          verbose:
        '''
        # Set up explainer model.
        explainer = self.explainer
        num_players = self.num_players
        imputer = self.imputer
        link = self.link
        normalization = self.normalization
        explainer.train()
        device = next(explainer.parameters()).device

        # Verify other arguments.
        if validation_samples is None:
            validation_samples = num_samples

        # Convert data.
        x_train = train_data
        x_val = val_data
        if isinstance(x_train, np.ndarray):
            x_train = torch.tensor(x_train, dtype=torch.float32)
            x_val = torch.tensor(x_val, dtype=torch.float32)
        elif isinstance(x_train, torch.Tensor):
            pass
        else:
            raise ValueError('data must be np.ndarray or torch.Tensor')

        # Grand coalition value.
        grand_train = calculate_grand_coalition(
            x_train, imputer, batch_size * num_samples, link, device).cpu()
        grand_val = calculate_grand_coalition(
            x_val, imputer, batch_size * num_samples, link, device).cpu()

        # Null coalition.
        with torch.no_grad():
            zeros = torch.zeros(1, num_players, dtype=torch.float32,
                                device=device)
            null = link(imputer(x_train[:1].to(device), zeros))
            if len(null.shape) == 1:
                null = null.reshape(1, 1)
        self.null = null

        # Generate validation data.
        sampler = ShapleySampler(num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        val_S = sampler.sample(validation_samples * len(x_val),
                               paired_sampling=True)
        x_val_tiled = x_val.unsqueeze(1).repeat(
            1, validation_samples, 1).reshape(
            len(x_val) * validation_samples, -1)
        val_values = []
        with torch.no_grad():
            for i in range(int(np.ceil(len(val_S) / batch_size))):
                start = i * batch_size
                end = min(start + batch_size, len(val_S))
                val_values.append(link(imputer(
                    x_val_tiled[start:end].to(device),
                    val_S[start:end].to(device))))
            val_values = torch.cat(val_values)

        # Set up train loader.
        train_set = TensorDataset(x_train, grand_train)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
            drop_last=True)

        # Set up val loader.
        val_S = val_S.reshape(
            len(x_val), validation_samples, num_players).cpu()
        val_values = val_values.reshape(
            len(x_val), validation_samples, -1).cpu()
        val_set = TensorDataset(x_val, grand_val, val_S, val_values)
        val_loader = DataLoader(val_set, batch_size=batch_size * num_samples,
                                pin_memory=True)

        # Setup for training.
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(explainer.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)
        self.loss_list = []
        best_loss = np.inf
        best_epoch = -1
        best_model = None
        if training_seed is not None:
            torch.manual_seed(training_seed)

        for epoch in range(max_epochs):
            # Sample minibatches.
            for x, grand in train_loader:
                # Sample S.
                S = sampler.sample(batch_size * num_samples,
                                   paired_sampling=paired_sampling)

                # Move to device.
                x = x.to(device)
                S = S.to(device)
                grand = grand.to(device)

                # Evaluate value function.
                x_tiled = x.unsqueeze(1).repeat(1, num_samples, 1).reshape(
                    batch_size * num_samples, -1)
                with torch.no_grad():
                    values = link(imputer(x_tiled, S))

                # Evaluate explainer.
                pred = explainer(x)
                pred = pred.reshape(batch_size, num_players, -1)

                # Efficiency penalty.
                if eff_lambda:
                    penalty = loss_fn(pred.sum(dim=1), grand - null)

                # Apply normalization.
                if normalization:
                    pred = normalization(pred, grand, null)

                # Evaluate loss.
                S = S.reshape(batch_size, num_samples, num_players)
                values = values.reshape(batch_size, num_samples, -1)
                approx = null + torch.matmul(S, pred)
                loss = loss_fn(approx, values)
                if eff_lambda:
                    loss = loss + eff_lambda * penalty

                # Take gradient step.
                loss = loss * num_players
                loss.backward()
                optimizer.step()
                explainer.zero_grad()

            # Evaluate validation loss.
            explainer.eval()
            val_loss = num_players * validate(
                val_loader, imputer, explainer, null, link,
                normalization).item()
            explainer.train()

            # Save loss, print progress.
            scheduler.step(val_loss)
            self.loss_list.append(val_loss)
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Val loss = {:.8f}'.format(val_loss))
                print('')

            # Check for convergence.
            if self.loss_list[-1] < best_loss:
                best_loss = self.loss_list[-1]
                best_epoch = epoch
                best_model = deepcopy(explainer)
                if verbose:
                    print('New best epoch, loss = {:.8f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early at epoch = {}'.format(epoch))
                break

        # Copy best model.
        for param, best_param in zip(explainer.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        explainer.eval()

    def shap_values(self, x):
        '''
        Generate SHAP values.

        Args:
          x:
        '''
        # Data conversion.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError('data must be np.ndarray or torch.Tensor')

        # Ensure batch dimension.
        if len(x.shape) == 1:
            single_explanation = True
            x = x.reshape(1, -1)
        else:
            single_explanation = False

        # Ensure null coalition is calculated.
        device = next(self.explainer.parameters()).device
        if self.null is None:
            with torch.no_grad():
                zeros = torch.zeros(1, self.num_players, dtype=torch.float32,
                                    device=device)
                null = self.link(self.imputer(x[:1].to(device), zeros))
            if len(null.shape) == 1:
                null = null.reshape(1, 1)
            self.null = null

        # Generate explanations.
        x = x.to(device)
        with torch.no_grad():
            pred = self.explainer(x)
            pred = pred.reshape(len(x), self.num_players, -1)
            if self.normalization:
                grand = calculate_grand_coalition(
                    x, self.imputer, len(x), self.link, device)
                pred = self.normalization(pred, grand, self.null)

        if single_explanation:
            return pred[0].cpu().data.numpy()
        else:
            return pred.cpu().data.numpy()
