import numpy as np
from openml.datasets import get_dataset


def _split_data_in_train_val(x_data, y_data, n_val=10000, seed=57):
    """
    Split data into train and validation.

    Args:
        x_data (np.array): [n x p] Two dimensional array of the input data. Rows correspond to datapoints and columns
            corresponds to features.
        y_data (np.array): [n] Array of the targets corresponding to `x_data`.
        n_val (int, optional): The amount of datapoints to use for the validation dataset.
        seed (int, optional): Random seed. Defaults to 57.


    Returns:
        dict: Dictionary of the data splits.
    """
    # Load random number generator (rng)
    rng = np.random.default_rng(seed=seed)
    n = x_data.shape[0]
    random_indices = rng.permutation(n)  # Get the numbers [0, 1, ..., n-1] in a random order

    # Index the amount of random indices corresponding to the respective split size
    n_train = len(x_data) - n_val
    train_indices = random_indices[:n_train]
    val_indices = random_indices[n_train:]

    x_train = x_data[train_indices]
    x_val = x_data[val_indices]

    y_train = y_data[train_indices]
    y_val = y_data[val_indices]

    all_data = {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
    }

    return all_data


def load_mnist_data(scale_x_data=True):
    """
    Loads and returns the MNIST dataset, splitted in train, validation and test sets.

    Returns:
        dict: The dictionary of the datasplits.
        scale_x_data (bool): Whether or not to scale the x-data, which is done by dividing by 255.
    """
    mnist = get_dataset(554)
    x_data, y_data, _, _ = mnist.get_data(dataset_format="dataframe", target="class")
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()

    x_main = x_data[:60000]
    y_main = y_data[:60000].astype(int)
    x_test = x_data[60000:]
    y_test = y_data[60000:].astype(int)

    all_data = _split_data_in_train_val(x_data=x_main, y_data=y_main)

    all_data["x_test"] = x_test
    all_data["y_test"] = y_test.astype(int)

    if scale_x_data is True:
        all_data["x_train"] = all_data["x_train"].astype(np.float32) / 255
        all_data["x_val"] = all_data["x_val"].astype(np.float32) / 255
        all_data["x_test"] = all_data["x_test"].astype(np.float32) / 255

    return all_data


def integer_one_hot_encode(x_array, max_int=None):
    """
    One hot encodes x_array.
    This assumes that x_arrays only has integer, and that the max element + 1 is the amount of class.
    Therefore, the classes should probably have consequtive values from 0 to c - 1.

    Arguments:
        x_array (np.array): (n) array of values to be one-hot-encoded.
        max_int (int): Max int of class.

    Returns:
        one_hot_array (np.array): (n x c) array of one-hot-encoded data.
    """
    if max_int is None:
        max_int = x_array.max()
    one_hot_array = np.zeros((x_array.shape[0], max_int + 1))  # Initialize empty array
    one_hot_array[np.arange(x_array.shape[0]), x_array] = 1  # Index rows (arange) and columns (x_array)
    one_hot_array = one_hot_array.astype(np.uint8)
    return one_hot_array


def softmax(x_data):
    """
    Performs the softmax function on x_data in a numerically stable way.
    Shifts exponent with the highest value.

    Arguments:
        x_data (np.array): [n x c]-shaped array of `n` inputs and `c` classes.

    Returns:
        np.array: [n x c]-shaped array with softmax applied to `x_data`.
    """
    shifted_x = x_data - np.max(x_data, axis=1, keepdims=True)
    softmaxes = np.exp(shifted_x) / np.sum(np.exp(shifted_x), axis=1, keepdims=True)
    return softmaxes


def calculate_multiclass_cross_entropy(targets, predictions):
    """
    Returns multiclass cross entropy loss, also called the log loss.
    Predictions should be (n_c) sized probabilities and targets should be of integer values (not one-hot encoded).

    Arguments:
        targets (np.array): [n]-shaped array of true target values.
        predictions (np.array): [n x c]-shaped array of predictions.

    Returns:
        float: The cross entropy.
    """
    pass


def calculate_accuracy(predictions, targets):
    """
    Calculates the accuracy of predictions and labels
    This is defined as (number of correct guesses) / (number of guesses).

    Arguments:
        predictions (np.array): [n]-shaped array of predictions.
        targets (np.array): [n]-shaped array of true targets for the same inputs as the predictions.

    Returns:
        float: The calculated accuracy.
    """
    pass


class IdentityActivation:
    def __call__(self, x_data):
        return x_data

    def diff(self, x_data):
        return np.ones(x_data.shape)


class NeuralNetwork:
    """
    Class implementing vanilla fully connected neural network.
    It can be trained with stochastic gradient descent (SGD) with backpropagation.
    Backpropagation is not implemented as automatic differentiation, but as a
    backward pass function.

    By convention after http://neuralnetworksanddeeplearning.com/chap2.html,
    weight matrices have next layer as rows, and previous layers as columns [n(l) x n(l-1)]

    Notation and sizes of arrays:
    []-marks sizes of dimensions. "d" marks derivative.
    "*" Marks elementwise product. "@" marks matrix multiplication. "o" marks omuter product.

    n: Amount of (mini-batch) inputs
    m: Amount of features in the inputs
    c: Amount of classes to predict (nodes in output layer)
    n(l): Amount of nodes in layer l.
    L: Amount of total layers.
    C(y; p): Cost function (loss function)

    x: [n x m] (mini-batch) input array
    p: [n x c] predictions, output array
    y: [n x c] true targets

    w(l) [n(l) x n(l-1)] weights for layer l (note the order of rows and colums
    b(l) [n(l) x 1] bias for layer l.
    z(l) [n x n(l)] weighted sum in layer l, z(l) = w(l) @ a(l) + b(l) (b(l) is broadcasted)
    a(l) [n x n(l)] activations in layer l, activation function of z, a = s(z), a(l) = s(l)(z(l))
    s(l) [callable] activation function in layer l.

    delta term, denoted "del(l)". Partial derivative denoted with "d", dy/dx
    Definition of delta terms: del(l) = dC / dz(l)

    Backpropagation formulas (non-vectorized, over index j and k):
    del(L)_j = dC/dp_j * s'(z(L)_j)
    del(l)_j = s'(z(l)_j) sum_k (w(l+1)_[k, j] del(l+1)_k)
    dC/db(l)_j = del(l)_j
    dC/dw(l)_[k, j] = del(l)_k * a(l-1)_j

    Backpropagation formulas (vectorized over layers):
    del(L) [c] = dC/dp [c] * s'(z(L)) [c]
    del(l) [n(l)] = (w(l+1).T [n(l) x n(l+1)] @ del(l+1) [n(l+1)]) [n(l)] * s'(z(l)) [n(l)]
    dC/db(l) [n(l)] = del(l) [n(l)]
    dC/dw(l) [n(l) x n(l-1)]= del(l) [n(l)] o a(l-1) [n(l-1)]

    Backpropagation formulas (vectorized over layers and minibatch):
    del(L) [n x c] = dC/dp [n x c] * s'(z(L)) [n x c]
    del(l) [n x n(l)] = (w(l+1).T [n(l) x n(l+1)] del(l+1).T [n(l+1) x n)]).T [n x n(l)] * s'(z(l)) [n x n(l)]
    del(l) [n x n(l)] = (del(l+1) [n x n(l+1)] @ w(l+1) [n(l+1) x n(l)]) [n x n(l)] * s'(z(l)) [n x n(l)]
    Weight and bias updates are summed over the n inputs:
    dC/db(l) [n(l)] = del(l) [n x n(l)].sum(axis=0)
    dC/dw(l) [n(l) x n(l-1)] = del(l).T [n(l) x n] @ a(l-1) [n x n(l-1)]
    """

    def __init__(self, layer_sizes, activation_functions, initialization_method="normal", seed=57):
        """
        Sets class variables for layer sizes, the amount of layers and the activation functions.

        Args:
            layer_sizes (list of int): List of the amount of nodes for each layer. The first element should corresponds
                to the amount of features in the dataset, and the last should be the amount of classes.
                The middle elements will be the amount of nodes in the hidden layers.
            activation_functions (list of callable): List of activation function to use for each hidden layer.
                Should be a callable class implementing `diff()` which returns its derivative.
            initialization_method (str): How the weights should be initialize. Please see the doc-string for
                `_initialize_weights()` for more information.
            seed (int): Random seed to make results reproducible.
        """
        np.random.seed(seed=seed)
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self._initialize_weights(layer_sizes=layer_sizes, initialization_method=initialization_method)

    def _initialize_weights(self, layer_sizes, initialization_method):
        """
        Initializes weights based on the `initialization_method` argument. Biases are set to zero.

        Methods:
            "zeros": Sets all weights to zeros. Used for debugging, not good for training models.
            "ones": Sets all weights to ones. Used for debugging, not good for training models.
            "normal": Draw all weights from N(0, 1/sqrt(n_in)).

        Arguments:
            layers_sizes (list): List of int of nodes in each layer.
            initialization_method (str): Method to use, see above.
        """
        self.biases = [np.zeros((1, layer_sizes[i])) for i in range(1, len(layer_sizes))]

        method = initialization_method.lower().strip()
        self.weights = []

        if method == "zeros":
            for i in range(len(layer_sizes) - 1):
                self.weights.append(np.zeros((layer_sizes[i + 1], layer_sizes[i])))

        elif method == "ones":
            for i in range(len(layer_sizes) - 1):
                self.weights.append(np.ones((layer_sizes[i + 1], layer_sizes[i])))

        elif method == "normal":
            for i in range(len(layer_sizes) - 1):
                std = 1 / np.sqrt(layer_sizes[i])
                self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * std)

        else:
            message = 'Argument `method` must be in ["zeros", "ones", "normal"]. '
            message += f"Was {method}. "
            raise ValueError(message)

    def forward(self, x_data):
        """
        Feeds the data forward. Do not discretize the output (give logits values).
        For each layer, calculates the activations, and feeds forward.

        Arguments:
            x_data (np.array): [n x m] data to forward.

        Returns:
            activations (np.array): [n x c] array over logits outputs (activations of last layer).
        """
        pass

    def predict(self, x_data):
        """
        Predicts on data, outputs the classes predicted.

        Arguments:
            x_data (np.array): [n x p]-shaped data to predict on.

        Returns:
            preds (np.array): [n]-shaped array of predicted classes (not one-hot-encoded).
        """
        pass

    def _backprop(self, preds, targets):
        """
        Perform backpropagation, returns deltas for each layer.

        Arguments:
            preds (np.array): [b x c] predictied logits values.
            targets (np.array): [b x c] true target-values.

        Returns:
            deltas (list): List of deltas for each layer (except input layer).
        """
        deltas = []  # Add deltas to list backwards (insert at 0 when adding)
        # del(L) [n x c] = dC/dp [n x c] * s'(z(L)) [n x c]
        delta_L = preds - targets
        deltas.append(delta_L)
        for i in range(1, self.n_layers - 1):
            index = self.n_layers - i - 1
            prev_delta = deltas[0]  # First element is the previous layers delta
            # del(l) [n x n(l)] = (del(l+1) [n x n(l+1)] @ w(l+1) [n(l+1) x n(l)]) [n x n(l)] * s'(z(l)) [n x n(l)]
            delta = prev_delta @ self.weights[index]
            delta = delta * self.activation_functions[-(i + 1)].diff(self.weighted_sums[index - 1])
            deltas.insert(0, delta)
        return deltas

    def _sgd(self, deltas, eta, n_data):
        """
        Uses vanilla Stochastic Gradient Descent (SGD) to update parameters.

        Arguments:
            deltas (list): The delta values returned from _backprop.
            eta (float): Learning rate of the optimizer.
            n_data (int): Amount of datapoints used in minibatch
        """
        for i in range(self.n_layers - 1):
            # dC/db(l) [n(l)] = del(l) [n x n(l)].sum(axis=0)
            d_biases = deltas[i].sum(axis=0)  # Normal update term
            self.biases[i] -= (eta / n_data) * d_biases

            # dC/dw(l) [n(l) x n(l-1)] = del(l).T [n(l) x n] @ a(l-1) [n x n(l-1)]
            d_weights = deltas[i].T @ self.activations[i]  # Normal update term
            self.weights[i] -= (eta / n_data) * d_weights

    def _run_single_epoch(self, x_train, y_train, minibatch_size, eta):
        """
        Perform one epoch of training.

        Arguments:
            x_train (np.array): [n x p]-shaped input data of n inputs and p features.
            y_train (np.array): [n x c]-shaped array of one-hot-encoded true targets.
            eta (float): Learning rate.
            minibatch_size (int): Size of mini-batch for stocastich gradient descent (SGD).
        """
        b = minibatch_size
        n = x_train.shape[0]
        for i in range(int(n / b)):  # Loop over all the minibatches for SGD
            upper_index = np.min((n, (i + 1) * b))  # Round of in case batch-size does not match up with n.
            n_data = upper_index - i * b  # Should be b unless last iteration
            batch = x_train[i * b : upper_index]  # Index batch
            targets = y_train[i * b : upper_index]  # Index targets
            preds = self.forward(batch)  # Get logits (ouput-nodes) values, and save other values
            deltas = self._backprop(preds, targets)  # Get delta-error terms from backprop
            self._sgd(deltas, eta, n_data)  # Update parameters

    def _perform_evaluation(self, x_train, y_train, n_epoch, loss_func, accuracy_func, eval_set=None):
        """
        Perform evaluation of loss and optinal accuracy, on both train set and optinal evaluation.

        Arguments:
            x_train (np.array): [n x p]-shaped input data of n inputs and p features.
            y_train (np.array): [n]-shaped array of true targets as integers.
            n_epoch (int): The epoch that we are on.
            loss_func (callable): Used for calculating loss on train set and eval set if it is not `None`.
            accuracy_func (callable): Used for calculating accuracy on train set and eval set if it is not `None`.
            eval_set (tuple, optional): If not None, will calculate validation loss and accuracy after each epoch.
                Should be on the format `eval_set = (x_val, y_val)`, where `x_val` and `y_val` corresponds
                to `x_train` and `y_train`.
        """
        train_logits = self.forward(x_train)
        self.train_losses[n_epoch] = loss_func(y_train, train_logits)
        train_preds = self.predict(x_train)
        self.train_accuracies[n_epoch] = accuracy_func(y_train, train_preds)

        if eval_set is not None:  # Stats for validation set
            x_val, y_val = eval_set
            val_logits = self.forward(x_val)
            self.val_losses[n_epoch] = loss_func(y_val, val_logits)
            val_preds = self.predict(x_val)
            self.val_accuracies[n_epoch] = accuracy_func(y_val, val_preds)

            print(f"Train-loss: {self.train_losses[n_epoch]:.5f}, ", end="")
            print(f"Validation-loss: {self.val_losses[n_epoch]:.5f}. ", end="")
            print(f"Train-accuracy: {self.train_accuracies[n_epoch]:.5f}, ", end="")
            print(f"Validation-accuracy: {self.val_accuracies[n_epoch]:.5f}")

    def train(self, x_train, y_train, eta, n_epochs, loss_func, accuracy_func, minibatch_size=64, eval_set=None):
        """
        Trains network.

        Arguments:
            x_train (np.array): [n x p]-shaped input data of n inputs and p features.
            y_train (np.array): [n]-shaped array of true targets as integers.
            eta (float): Learning rate.
            n_epochs (int): The amount of epochs (iterations over all data) to train for.
            loss_func (callable): Used for calculating loss on train set and eval set if it is not `None`.
            accuracy_func (callable): Used for calculating accuracy on train set and eval set if it is not `None`.
            minibatch_size (int): Size of mini-batch for stocastich gradient descent (SGD).
            eval_set (tuple, optional): If not None, will calculate validation loss and accuracy after each epoch.
                Should be on the format `eval_set = (x_val, y_val)`, where `x_val` and `y_val` corresponds
                to `x_train` and `y_train`.
        """
        # Initialize losses and accuracies
        self.train_losses = np.zeros(n_epochs)
        self.train_accuracies = np.zeros(n_epochs)
        if eval_set is not None:
            self.val_losses = np.zeros(n_epochs)
            self.val_accuracies = np.zeros(n_epochs)

        y_train_one_hot = integer_one_hot_encode(y_train, max_int=9)
        for n_epoch in range(n_epochs):
            print(f"Epoch number [{n_epoch + 1} / {n_epochs}]")
            self._run_single_epoch(x_train=x_train, y_train=y_train_one_hot, minibatch_size=minibatch_size, eta=eta)
            self._perform_evaluation(
                x_train=x_train,
                y_train=y_train,
                loss_func=loss_func,
                accuracy_func=accuracy_func,
                n_epoch=n_epoch,
                eval_set=eval_set,
            )
