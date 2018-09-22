"""A model wrapper to perform Monte Carlo simulations."""
import numpy as np
from keras.models import Model
from tqdm import tqdm


class MonteCarlo(object):
    """A model wrapper to perform Monte Carlo simulations."""

    def __init__(self, model: Model, simulations: int) -> None:
        """
        Initialize a new Monte Carlo model wrapper.

        Args:
            model: the Bayesian model to estimate mean output using Monte Carlo
            simulations: the number of simulations to estimate mean

        Returns:
            None

        """
        # type check the model and store
        if not isinstance(model, Model):
            raise TypeError('model must be of type {}'.format(Model))
        self.model = model
        # type check the simulations parameter and store
        try:
            self.simulations = int(simulations)
        except ValueError:
            raise TypeError('simulations must be an integer')

    def _simulate(self, method, *args, **kwargs):
        # create a list to store the output predictions in
        simulations = [None] * self.simulations
        # evaluate over the number of simulations
        for idx in tqdm(range(self.simulations), unit='simulation'):
            simulations[idx] = getattr(self.model, method)(*args, **kwargs)
        # return the mean of each metric over the simulations
        return simulations

    @property
    def layers(self):
        return self.model.layers

    def get_layer(self, *args, **kwargs):
        """Retrieve a layer based on either its name (unique) or index."""
        return self.model.get_layer(*args, **kwargs)

    @property
    def updates(self):
        """Retrieve the model's updates."""
        return self.model.updates

    @property
    def losses(self):
        """Retrieve the model's losses."""
        return self.model.losses

    @property
    def uses_learning_phase(self):
        return self.model.uses_learning_phase

    @property
    def stateful(self):
        return self.model.stateful

    def reset_states(self):
        return self.model.reset_states()

    @property
    def state_updates(self):
        """Return the `updates` from all layers that are stateful."""
        return self.model.state_updates
    
    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights
        
    def get_weights(self):
        """Return the weights of the model."""
        return self.model.get_weights()

    def set_weights(self, weights: list):
        """
        Set the weights of the model.

        Args:
            weights: Numpy arrays matching output of `get_weights`

        Returns:
            None

        """
        return self.model.set_weights(weights)

    @property
    def input_spec(self):
        return self.model.input_spec

    def call(self, *args, **kwargs):
        raise ValueError('cannot call a Monte Carlo simulation Model.')

    def compile(self, *args, **kwargs):
        """Configure the model for training."""
        return self.model.compile(*args, **kwargs)
        
    @property
    def metrics_names(self) -> list:
        """Return the names of metrics for this Model wrapper."""
        return self.model.metrics_names

    def fit(self, *args, **kwargs):
        """Train the model for a given number of epochs (iterations on a dataset)."""
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> list:
        """Return the loss value & metrics values for the model in test mode."""
        # simulate the method
        metrics = self._simulate('evaluate', *args, **kwargs)
        # return the mean of each metric over the simulations
        return np.mean(metrics, axis=0)

    def predict(self, *args, **kwargs) -> np.ndarray:
        """
        Return mean target and output variance for given inputs.

        Args:
            args: the positional arguments for evaluate_generator
            kwargs: the keyword arguments for evaluate_generator
            
        Returns:
            a tuple of:
            - mean predictions over self.simulations passes
            - variance of predictions over self.simulations passes

        """
        # simulate the method
        y = self._simulate('predict', *args, **kwargs)
        # return the mean and variance of the simulations
        return np.mean(y, axis=0), np.var(y, axis=0)

    def train_on_batch(self, *args, **kwargs):
        """Runs a single gradient update on a single batch of data."""
        return self.model.train_on_batch(*args, **kwargs)

    def test_on_batch(self, *args, **kwargs):
        """Test the model on a single batch of samples."""
        # simulate the method
        metrics = self._simulate('test_on_batch', *args, **kwargs)
        # return the mean of each metric over the simulations
        return np.mean(metrics, axis=0)

    def predict_on_batch(self, *args, **kwargs):
        """Returns predictions for a single batch of samples."""
        # simulate the method
        y = self._simulate('predict_on_batch', *args, **kwargs)
        # return the mean and variance of the simulations
        return np.mean(y, axis=0), np.var(y, axis=0)

    def fit_generator(self, *args, **kwargs):
        """Train the model on data generated batch-by-batch."""
        return self.model.fit_generator(*args, **kwargs)

    def evaluate_generator(self, *args, **kwargs) -> list:
        """
        Evaluate the model using a generator for input.

        Args:
            args: the positional arguments for evaluate_generator
            kwargs: the keyword arguments for evaluate_generator
            
        Returns:
            the mean evaluation metrics over self.simulations passes

        """
        # simulate the method
        metrics = self._simulate('evaluate_generator', *args, **kwargs)
        # return the mean of each metric over the simulations
        return np.mean(metrics, axis=0)

    def predict_generator(self, *args, **kwargs):
        """Generate predictions for the input samples from a data generator."""
        # simulate the method
        y = self._simulate('predict_generator', *args, **kwargs)
        # return the mean and variance of the simulations
        return np.mean(y, axis=0), np.var(y, axis=0)


# explicitly define the outward facing API of this module
__all__ = [MonteCarlo.__name__]
