from scipy.special import expit

import numpy as np
import os


def mean_squared_error(prediction, target, exclude_diagonal=False):
    """Calculate the mean squared difference between two numpy arrays

    :param prediction: Predicted values
    :param target: Target values
    :param exclude_diagonal: If True, ignore the entries on the diagonal of the matrices
    :return: The mean squared difference between prediction and target
    """
    squared_difference = (prediction - target) ** 2
    if exclude_diagonal:
        np.fill_diagonal(squared_difference, 0.)
    return np.mean(squared_difference)


def fit_sigmoid(prediction, target, initial_s=1., epsilon=1e-8, exclude_diagonal=False):
    """Find the slope of a sigmoidal that minimises the mean squared error between a prediction and a target numpy array

    :param prediction: Predicted values
    :param target: Target values
    :param initial_s: Initial values of the slope parameter to start the iterative fitting process from
    :param epsilon: Stopping criterion
    :param exclude_diagonal: If True, ignore the entries on the diagonal of the matrices
    :return: The slope that minimises
    """
    s = initial_s
    delta_s = .1
    last_mse = mean_squared_error(expit(s * prediction), target, exclude_diagonal)
    while np.abs(delta_s) > epsilon:
        mse_ = mean_squared_error(expit((s + delta_s) * prediction), target, exclude_diagonal)
        if last_mse == mse_:
            break
        if last_mse < mse_:
            delta_s = (-1. * delta_s) / 2.
        s += delta_s
        last_mse = mse_
    return s


def rotate(xs, ys, angle):
    """Rotate a set of x and y values by an angle

    :param xs: Numpy array of x values
    :param ys: Numpy array of y values
    :param angle: Rotation angle
    """
    angle = angle * np.pi / 180.
    for i, (x, y) in enumerate(zip(xs, ys)):
        xs[i] = x * np.cos(angle) - y * np.sin(angle)
        ys[i] = x * np.sin(angle) + y * np.cos(angle)


def load_behavioural_data(directory=""):
    """Load experimental behavioural data

    :param directory: Directory of behavioural files
    :return: Behavioural data
    """
    midd_performance = np.loadtxt(os.path.join(directory, "test-short.txt"), delimiter=",")
    high_performers = np.loadtxt(os.path.join(directory, "test-long-high-performers.txt"), delimiter=",")
    low_performers = np.loadtxt(os.path.join(directory, "test-long-low-performers.txt"), delimiter=",")

    # Scale to range [0., 1.]
    high_performers -= 1.
    low_performers -= 1.
    return midd_performance, high_performers, low_performers


def softmax(x):
    """Calcualte softmax function

    :param x: Logits
    :return: Probabilities
    """
    return np.exp(x) / np.sum(np.exp(x))


def response_time(x, threshold):
    """Calculate response times from trajectory

    :param x: Trajectory
    :param threshold: Threshold from race condition
    :return: Response time
    """
    rt = np.argmax(x > threshold)
    return np.nan if rt == 0 else rt
