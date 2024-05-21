import torch
import torch.nn as nn


class CAModel(nn.Module):
    """Cellular Automata Model
    
    Parameters
    ----------
    n_channels : int
        Number of channels of the grid.
        
    hidden_channels : int
        Hidden channels relates to pixel wise 2d conv
        
    fire_rate : float
        Number between 0 and 1. The lower it is the more likely cells will be set to zero
        
    device : torch.device
        Determines the device preforming the compute
    
    Attributes
    ----------
    update_module : nn.Sequential
        The only part 
    
    filters : torch.Tensor

    """
    def __init__(self, n_channels=16, hidden_channels=128, fire_rate=0.5, device=None):
        super().__init__()
        
        self.fire_rate = fire_rate
        self.n_channels = n_channels
        self.device = device or torch.device("cpu")
        self.hidden_channels = hidden_channels
        
        # Perceive step
        sobel_filter_ = torch.tensor([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ])
        scalar = 8.0
        
        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        
        identity_filter = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=torch.float32)
        
        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y])
        
        filters = filters.repeat((n_channels, 1, 1))
        
        self.filters = filters[:, None, ...].to(
            self.device
        )
        
        
        self.update_module = nn.Sequential(
            nn.Conv2d(
                3*n_channels,
                hidden_channels,
                kernel_size=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                n_channels,
                kernel_size=1,
                bias=False,
            ),
        )
        
        with torch.no_grad():
            self.update_module[2].weight.zero_()
        
        self.to(self.device)
        
    def perceive(self, x):
        """Approximate channelwise gradient and combine with the input.
        
        This is the only place where we include information on the 
        neighboring cells. However, we are not using any learnable 
        parameter here.
        
        Parameter
        ---------
        x : torch.Tensor
            Shape '(n_samples, n_channels, grid_size, grid_size)'
            
        Returns
        -------
        torch.Tensor
            Shape '(n_samples, 3*n_channels, grid_size, grid_size)'
        """
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)
    
    def update(self, x):
        """Perform update.
        
        Note that this is the only part of the forward pass that uses 
        trainable parameters.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape '(n_samples, 3*n_channels, grid_size, grid_size)'
            
        Returns
        -------
        torch.Tensor
            Shape '(n_samples, n_channels, grid_size, grid_size)'
        """
        return self.update_module(x)
        
    @staticmethod
    def stochastic_update(x, fire_rate):
        """Run pixel-wise dropout.
        
        Unlike dropout there is no scaling taking place.
        
        Parameters
        ----------
        x : tensor.Tensor
            Shape '(n_samples, n_channels, grid_size, grid_size)'
        
        fire_rate : float
            Number between 0 and 1. The lower it is the more likely cells will be set to zero
            
        Returns
        -------
        torch.Tensor
            Shape '(n_samples, n_channels, grid_size, grid_size)'
        """
        device = x.device
        
        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask
    
    @staticmethod
    def get_living_mask(x):
        """Identify living cells
        
        Parameters
        ----------
        x : tensor.Tensor
            Shape '(n_samples, n_channels, grid_size, grid_size)'
        
        Returns
        -------
        torch.Tensor
            Shape '(n_samples, 1, grid_size, grid_size)' ans the
            dtype is bool
        """
        return (
            nn.functional.max_pool2d(
                x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1
            ) > 0.1
        )
        
    def forward(self, x):
        """Run the forward pass.
        
        Parameters
        ----------
        x : tensor.Tensor
            Shape '(n_samples, n_channels, grid_size, grid_size)'
        
        Returns
        -------
        torch.Tensor
            Shape '(n_samples, n_channels, grid_size, grid_size)'
        """
        pre_life_mask = self.get_living_mask(x)
        
        y = self.perceive(x)
        dx = self.update(y)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)
        
        x = x + dx
        
        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        
        return x * life_mask
    