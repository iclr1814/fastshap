import torch
import numpy as np


class MarginalImputer:
    '''
    Evaluate a model while replacing features with samples from the marginal
    distribution.

    Args:
      model:
      background:
      groups:
    '''
    def __init__(self, model, background, groups=None):
        self.model = model
        self.background = background
        self.background_repeat = background
        self.n_background = len(background)

        # Store feature groups.
        num_features = background.shape[1]
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
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

    def __call__(self, x, S):
        '''
        Evaluate model with marginal imputation.

        Args:
          x:
          S:
        '''
        # Set up background repeat.
        if len(self.background_repeat) != len(x) * self.n_background:
            self.background_repeat = np.tile(self.background, (len(x), 1))

        # Prepare x and S.
        if isinstance(x, torch.Tensor):
            torch_conversion = True
            device = x.device
            x = x.cpu().data.numpy()
            S = S.cpu().data.numpy()
        else:
            torch_conversion = False
        x = x.repeat(self.n_background, 0)
        S = S.astype(bool)
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)
        S = S.repeat(self.n_background, 0)

        # Replace features.
        x_ = x.copy()
        x_[~S] = self.background_repeat[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.n_background, *pred.shape[1:])
        pred = np.mean(pred, axis=1)
        if torch_conversion:
            pred = torch.tensor(pred, dtype=torch.float32, device=device)
        return pred


class BaselineImputer:
    '''
    Evaluate a model while replacing features with baseline values.

    Args:
      model:
      baseline:
      groups:
    '''
    def __init__(self, model, baseline, groups=None):
        self.model = model
        self.baseline = baseline
        self.baseline_repeat = baseline

        # Store feature groups.
        num_features = baseline.shape[1]
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
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

    def __call__(self, x, S):
        '''
        Evaluate model with baseline values.

        Args:
          x:
          S:
        '''
        # Prepare x and S.
        if isinstance(x, torch.Tensor):
            torch_conversion = True
            device = x.device
            x = x.cpu().data.numpy()
            S = S.cpu().data.numpy()
        else:
            torch_conversion = False
        S = S.astype(bool)
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Prepare baseline repeat.
        if len(self.baseline_repeat) != len(x):
            self.baseline_repeat = self.baseline.repeat(len(x), 0)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.baseline_repeat[~S]

        # Make predictions.
        pred = self.model(x_)
        if torch_conversion:
            pred = torch.tensor(pred, dtype=torch.float32, device=device)
        else:
            pred = pred.astype(np.float32)
        return pred
