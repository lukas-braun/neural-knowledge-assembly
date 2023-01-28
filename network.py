import torch
import numpy as np
from scipy.special import expit
from scipy.integrate import odeint


class Network(torch.nn.Module):
    def __init__(self, items_n, h1_size, w1_weight_std, w2_weight_std, non_linearity=torch.relu_, readouts=1):
        """Two-layer neural network to calculate the pairwise relationships of two items. In particular, the network
        function is W_2 sigma(W_1x_1 - W_1x_2), where x_1 and x_2 are one-hot vectors that indicate the index of the
        presented item. The network either has one or two readout heads to either jointly or independently encode the
        relationship between items.

        :param items_n: Number of items
        :param h1_size: Size of hidden layer
        :param w1_weight_std: Standard deviation of initial weights of input layer
        :param w2_weight_std: Standard deviation of initial weights of readout layer
        :param non_linearity: Non-linearity to be applied on the hidden layer representation
        :param readouts: Number of readout heads (1 or 2)
        """
        super().__init__()
        self.items_n = items_n
        self.h1_size = h1_size
        self.non_linearity = non_linearity
        self.readouts = readouts

        self.pairwise_certainty = PairwiseCertainty(items_n)

        self.loss = None
        self.item_1 = None
        self.item_2 = None

        # Create and initialise network layers
        self.layer_1 = torch.nn.Linear(items_n, h1_size, bias=False)
        self.layer_2 = torch.nn.Linear(h1_size, readouts, bias=False)
        torch.nn.init.normal_(self.layer_1.weight, std=w1_weight_std)
        torch.nn.init.normal_(self.layer_2.weight, std=w2_weight_std)

    def forward(self, item_1, item_2):
        """Calculate the network output

        :param item_1: Index of the first item
        :param item_2: Index of the second item
        :return:
        """
        self.item_1 = item_1
        self.item_2 = item_2

        x1 = self._one_hot(item_1)
        x2 = self._one_hot(item_2)
        h1 = self.non_linearity(self.layer_1(x1) - self.layer_1(x2))

        out = self.layer_2(h1)
        return h1, out

    def correct(self, learning_rate, gamma):
        """Preserve previously learned relationships between items dependent on the certainty that two items are
        correctly related in embedding space.

        :param learning_rate: Learning rate of gradient descent
        :param gamma: Time-constant of low-pass filter to acquire certainties
        :return:
        """
        self.pairwise_certainty.update(self.item_1, self.item_2, self.loss.item(), gamma)
        items = [self.item_1, self.item_2]

        with torch.no_grad():
            for item, other_item in zip(items, items[::-1]):
                # Calculate relative changes of weights
                w1 = self.layer_1.weight
                dw1 = -learning_rate * self.layer_1.weight.grad[:, item]
                w2 = self.layer_2.weight[0]
                dw2 = -learning_rate * self.layer_2.weight.grad[0]

                if torch.linalg.norm(dw2) > 1e-6 and torch.linalg.norm(dw1) > 1e-6:
                    # Apply corrections
                    dw1_ = dw1
                    nominator = (w2 @ dw1_ + dw2 @ w1[:, item] + dw2 @ dw1_ - dw2 @ w1)
                    denominator = (w2 @ dw1_ + dw2 @ dw1_)
                    cs = nominator / denominator
                    cs[item] = 0.
                    cs[other_item] = 0.
                    certainty = torch.tensor(self.pairwise_certainty.a[:, item])
                    w1 += torch.outer(dw1_, certainty * cs)

    def evaluate(self):
        """Evaluate network output on all possible combinations of input items

        :return: Numpy array of size (items_n, items_n) containing all network outputs
        """
        with torch.no_grad():
            n = self.items_n
            grid = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    with torch.no_grad():
                        if self.readouts == 1:
                            grid[j, i] = self.forward(i, j)[1].item()
                        elif self.readouts == 2:
                            grid[j, i] = self.forward(i, j)[1][0].item() - self.forward(i, j)[1][1].item()
            return grid

    def extract_h1s(self):
        """Calculate network hidden state representations for all input items

        :return: Numpy array of size (items_n, h1_size) containing all hidden states
        """
        with torch.no_grad():
            n = self.items_n
            h1s = np.zeros((n, self.h1_size))
            for i in range(n):
                with torch.no_grad():
                    h1s[i, :] = self.layer_1.weight[:, i].detach().numpy().copy()
            return h1s

    def _one_hot(self, item):
        """ Create one-hot vector from index

        :param item: Item index
        :return: One-hot vector
        """
        x = torch.zeros(self.items_n)
        x[item] = 1.
        return x


class PairwiseCertainty:
    def __init__(self, items_n, a=-1000., b=0.01):
        """Object to tract pairwise certainties

        :param items_n: Number of items
        :param a: Slope of sigmoidal
        :param b: Offset of sigmoidal
        """
        self.a = np.zeros((items_n, items_n))
        self.slope = a
        self.offset = b

    def update(self, i1, i2, loss, gamma):
        """Update certainty matrix based on performance of items a and b

        :param i1: Index of first item
        :param i2: Index of second item
        :param loss: Current loss value
        :param gamma: Time-constant of low-pass filter
        """
        certainty = PairwiseCertainty.phi(loss, self.slope, self.offset)
        a = self.a
        a_ = a.copy()

        a[i1, :] = a[:, i1] = (1. - gamma) * a_[i1, :] + gamma * certainty * a_[i2, :]
        a[i2, :] = a[:, i2] = (1. - gamma) * a_[i2, :] + gamma * certainty * a_[i1, :]
        a[i1, i2] = a[i2, i1] = (1. - gamma) * a_[i1, i2] + gamma * certainty

    @staticmethod
    def phi(x, a, b):
        """Calculate certainty

        :param x: Current loss value
        :param a: Slope of sigmoidal
        :param b: Offset of sigmoidal
        :return: Certainty value
        """
        return expit(a * (x - b))


class Remerge:
    """Implementation of the REMERGE model (Kumaran McClelland, 2012)"""
    def __init__(self, inputs_n):
        """Initialise REMERGE model for a set of inputs

        :param inputs_n: Number of inputs
        """
        self.inputs_n = inputs_n
        self.hidden_size = inputs_n - 1
        self.W, self.W_out = self._init_ws()
        self.is_stitched(False)

    def is_stitched(self, stitched):
        """Change connectivity matrix to assembled mode

        :param stitched: Boolean which indicates if weight matrix should in assembled mode or not
        """
        i_half = self.inputs_n // 2
        if not stitched:
            self.W[i_half - 1, :] = 0.
            self.W_out[:, i_half - 1] = 0.
        else:
            self.W[i_half - 1, [i_half - 1, i_half]] = 0.5
            self.W_out[i_half, i_half - 1] = 1.
            self.W_out[i_half - 1, i_half - 1] = -1.

    def run(self, item_1, item_2, ts):
        """Unroll dynamics and calculate network prediction for a pair of items

        :param item_1: Index of item 1
        :param item_2: Index of item 2
        :param ts: Time to evaluate the model on
        :return: Hidden states and network outputs
        """
        u = np.zeros(self.inputs_n)
        u[item_1] = 1.
        if item_2 >= 0:
            u[item_2] = 1.

        y0 = np.zeros(self.inputs_n + self.hidden_size)
        dynamics = odeint(self.dynamics, y0, ts, (self.W, u, self.inputs_n))

        hs = dynamics[:, self.inputs_n:]

        predictions = self.W_out @ hs.T
        return hs, predictions

    @staticmethod
    def dynamics(y, t, W, u, inputs_n):
        """Calculate network dynamics from a set of parameters

        :param y: Inputs and hidden states
        :param t: Point in time
        :param W: Weight matrix
        :param u: Inputs
        :param inputs_n: Number of inputs
        :return: Inputs and hidden state of network
        """
        x = y[:inputs_n]
        h = y[inputs_n:]

        dx = -x + W.T @ h + u
        dh = -h + W @ x

        return np.concatenate([dx, dh])

    def _init_ws(self):
        """Initialise the weight matrices of the REMERGE model

        :return: Recurrent input to hidden and output weights
        """
        W = np.zeros((self.hidden_size, self.inputs_n))
        np.fill_diagonal(W, 1)
        rng = np.arange(self.hidden_size)
        W[rng, rng + 1] = 1.
        W *= 0.5

        W_out = np.zeros((self.inputs_n, self.hidden_size))
        np.fill_diagonal(W_out, 1)
        rng = np.arange(1, self.inputs_n)
        W_out[rng, rng - 1] = -1.
        W_out *= -1.

        return W, W_out
