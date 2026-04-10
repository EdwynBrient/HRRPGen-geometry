import torch
from math import pi
import torch.nn as nn

class QuadraticScheduler:
    clip_max_value = torch.Tensor([0.999])

    def __init__(self, T: int, max_beta: float = 0.02):
        """
        Quadratic variance scheduler.
        The equation for the variance is:
            beta = min(max_beta, (t/T)^2)
        The equation for the alpha is:
            alpha = 1 - beta
        The equation for the beta_hat is:
            beta_hat = (1 - alpha_hat(t - 1)) / (1 - alpha_hat(t)) * beta(t)
        """
        self.T = T
        self.max_beta = max_beta
        self._betas = self._beta_function(torch.arange(self.T), T, max_beta)
        self._alphas = 1.0 - self._betas
        self._alpha_hats = torch.cumprod(self._alphas, dim=0)
        self._alpha_hats_t_minus_1 = torch.roll(self._alpha_hats, shifts=1, dims=0)
        self._alpha_hats_t_minus_1[0] = self._alpha_hats_t_minus_1[1]  # to remove first NaN value
        self._betas_hat = (1 - self._alpha_hats_t_minus_1) / (1 - self._alpha_hats) * self._betas
        self._betas_hat[torch.isnan(self._betas_hat)] = 0.0

    def _beta_function(self, t: torch.Tensor, T: int, max_beta: float):
        """
        Compute the beta value for each t using a quadratic schedule.
        :param t: the t values
        :param T: the total number of noising steps
        :param max_beta: the maximum beta value
        """
        beta_values = torch.minimum(max_beta, (t / T) ** 2)
        return beta_values

    def get_alpha_hat(self):
        return self._alpha_hats

    def get_alphas(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_betas_hat(self):
        return self._betas_hat


class CosineScheduler(nn.Module):
    def __init__(self, T: int, s: float = 0.008, clipping_value: float = 0.999):
        super(CosineScheduler, self).__init__()
        """
        Cosine variance scheduler.
        The equation for the variance is:
            alpha_hat = min(cos((t / T + s) / (1 + s) * pi / 2)^2, 0.999)
        The equation for the beta is:
            beta = 1 - (alpha_hat(t) / alpha_hat(t - 1))
        The equation for the beta_hat is:
            beta_hat = (1 - alpha_hat(t - 1)) / (1 - alpha_hat(t)) * beta(t)
        """
        self.T = T
        self.clipping_value = torch.Tensor([clipping_value])
        self._alpha_hats = self._alpha_hat_function(torch.arange(self.T), T, s)
        self._alpha_hats_t_minus_1 = torch.roll(self._alpha_hats, shifts=1, dims=0) # shift forward by 1 so that alpha_first[t] = alpha[t-1]
        self._alpha_hats_t_minus_1[0] = self._alpha_hats_t_minus_1[1]  # to remove first NaN value
        self._betas = 1.0 - self._alpha_hats / self._alpha_hats_t_minus_1
        self._betas = torch.minimum(self._betas, self.clipping_value)
        self._alphas = 1.0 - self._betas
        self._betas_hat = (1 - self._alpha_hats_t_minus_1) / (1 - self._alpha_hats) * self._betas
        self._betas_hat[torch.isnan(self._betas_hat)] = 0.0

    def forward(self):
        return self._betas

    def _alpha_hat_function(self, t: torch.Tensor, T: int, s: float):
        """
        Compute the alpha_hat value for a given t value.
        :param t: the t value
        :param T: the total amount of noising steps
        :param s: smoothing parameter
        """
        cos_value = torch.pow(torch.cos((t / T + s) / (1 + s) * pi / 2.0), 2)
        return cos_value

    def get_alpha_hat(self):
        return self._alpha_hats

    def get_alphas(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_betas_hat(self):
        return self._betas_hat

class MonotonicDense(nn.Module):
    def __init__(self, in_features, out_features):
        """
        A dense layer with weights constrained to be non-negative for monotonic behavior.
        Args:
            in_features (int): Input dimensionality.
            out_features (int): Output dimensionality.
        """
        super(MonotonicDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and biases
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.biases = nn.Parameter(torch.zeros(out_features))

        self.weights.data.fill_(0.02)  # Set weights proportional to the slope
        self.biases.data.fill_(0.0001)  # Set biases as the intercept

    def forward(self, x):
        # Apply ReLU to constrain weights to be non-negative
        positive_weights = torch.relu(self.weights)
        return torch.matmul(x, positive_weights.T) + self.biases

class LearnableScheduler(nn.Module):
    def __init__(self, T, hidden_dim=128, clipping_value=0.999):
        """
        A learnable noise scheduler with additional functionalities like alpha_hat, beta, and beta_hat.
        Args:
            T (int): Number of timesteps.
            hidden_dim (int): Dimension of the hidden layer for the MLP.
            clipping_value (float): Maximum value for clipping beta.
        """
        super(LearnableScheduler, self).__init__()
        self.T = T
        self.clipping_value = clipping_value
        self.lbd=torch.Tensor([1.0])

        # Simple MLP to predict beta_t
        self.scheduler = nn.Sequential(
            MonotonicDense(1, hidden_dim),  # Input is the normalized timestep
            MonotonicDense(hidden_dim, hidden_dim),
            MonotonicDense(hidden_dim, 1),
            nn.Sigmoid()  # Outputs beta in the range [0, 1]
        )

        # Initialize linearscheduler
        self.linear_betas = 0.0001 + (0.02 - 0.0001) * torch.linspace(0, 400, T) / T

        # Initialize timesteps
        self.register_buffer("timesteps", torch.linspace(0, 1, T).view(-1, 1))  # Normalize t

        # Initialize beta values
        self._betas = None
        self._alphas = None
        self._alpha_hats = None
        self._betas_hat = None

    def forward(self):
        """
        Compute the beta schedule for all timesteps.
        Returns:
            betas (torch.Tensor): Beta values for all timesteps.
        """
        # Pass normalized timesteps through the scheduler
        betas = self.scheduler(self.timesteps)
        # Clip beta values
        betas = torch.minimum(betas, torch.tensor(self.clipping_value, device=betas.device))
        self._betas = betas.squeeze()
        self._betas = self.lbd.to(self._betas.device) * self.linear_betas.to(self._betas.device) + (1 - self.lbd.to(self._betas.device)) * self._betas
        self.update_lbd()
        return self._betas

    def update_lbd(self):
        self.lbd = self.lbd * 0.99

    def compute_alpha_hat(self):
        """
        Compute alpha_hat (cumulative product of (1 - beta)).
        """
        if self._betas is None:
            self.forward()  # Ensure betas are computed
        alphas = 1.0 - self._betas
        self._alphas = alphas
        self._alpha_hats = torch.cumprod(alphas, dim=0)

    def compute_betas_hat(self):
        """
        Compute beta_hat for all timesteps.
        beta_hat = (1 - alpha_hat(t-1)) / (1 - alpha_hat(t)) * beta(t)
        """
        if self._alpha_hats is None:
            self.compute_alpha_hat()
        # Shift alpha_hats by 1 timestep
        alpha_hats_t_minus_1 = torch.roll(self._alpha_hats, shifts=1, dims=0)
        alpha_hats_t_minus_1[0] = alpha_hats_t_minus_1[1]  # Handle the first timestep
        self._betas_hat = (1 - alpha_hats_t_minus_1) / (1 - self._alpha_hats) * self._betas
        self._betas_hat[torch.isnan(self._betas_hat)] = 0.0

    def get_alpha_hat(self):
        if self._alpha_hats is None:
            self.compute_alpha_hat()
        return self._alpha_hats

    def get_alphas(self):
        if self._alphas is None:
            self.compute_alpha_hat()
        return self._alphas

    def get_betas(self):
        if self._betas is None:
            self.forward()
        return self._betas

    def get_betas_hat(self):
        if self._betas_hat is None:
            self.compute_betas_hat()
        return self._betas_hat

def inverse_warmup_lr_lambda(current_step: int, warmup_steps: int = 100, initial_lr_factor: float = 10):
    """
    Learning rate schedule that starts with a high LR and decreases to the base LR during warm-up.
    Args:
        current_step: Current step in training.
        warmup_steps: Total number of warm-up steps.
        initial_lr_factor: The factor by which the initial LR is scaled (e.g., 10 means 10x the base LR).
    """
    if current_step < warmup_steps:
        scale = initial_lr_factor - (initial_lr_factor - 1) * (current_step / float(warmup_steps))
        return scale
    return 1.0
