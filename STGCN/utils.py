import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import savefig
from tensorflow import keras
from tqdm import tqdm
from eval_utils import evaluation


# Used to predict the next N steps in traffic
def eval_next_steps(model, data, n_history, n_pred, data_stats):

    predictions = []
    ground_truth = np.expand_dims(data[:,-n_pred:], -1)

    print("\nEvaluating...")
    # Predict the next n_pred steps
    for i in tqdm(range(0, n_pred)):

        # Apparently STGCN does not use its own predictions during inference
        input_data = data[:,i:i+n_history]
        input_data = np.expand_dims(input_data, -1)
        pred = model.predict(input_data)
        predictions.append(pred)

    # Concatenate our predictions
    predictions = np.concatenate(predictions, axis=1)

    # Evaluate our predictions
    evaluation(ground_truth, predictions, data_stats)


# To run at the end of every epoch to evaluate our trained model
class evalCallback(keras.callbacks.Callback):
    # Initialize the callback with data and our shapes (e.g. how much we predict)
    def __init__(self, val_data, n_history, n_pred, model, data_stats):
        self.val_data = val_data
        self.n_history = n_history
        self.n_pred = n_pred
        self.data_stats = data_stats

    # At the end of the epoch, predict the next n_pred steps
    def on_epoch_end(self, epoch, logs):
        eval_next_steps(self.model, self.val_data, self.n_history, self.n_pred, self.data_stats)


# Perform weight computation from distance matrix
def weight_computation(distance_matrix):
  sigma2 = 0.1
  epsilon = 0.5
  n = distance_matrix.shape[0]
  weight_matrix = distance_matrix / 10000.
  W2, W_mask = weight_matrix * weight_matrix, np.ones([n, n]) - np.identity(n)
  weight_matrix = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

  return weight_matrix

# Perform a rolling window over all our data
#  Should create (num_datapoints - width, width, num_stations)
def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    return np.array([ a[i:i+width] for i in range(0,n-width, stepsize)])

def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


# Create train val test set
# Basically, takes a rolling window of frame_size across all_data, and splits it
#  according to train validation and test proportions.
# It's pretty much what STGCN does, except they first chop into train/val/test before
#  doing the rolling window
def create_train_val_test(all_data, frame_size, split_config):

    train_ratio = split_config["train"]
    val_ratio = split_config["val"]
    test_ratio = split_config["test"]

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    data_stats = {'mean': np.mean(all_data), 'std': np.std(all_data)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    all_data = z_score(all_data, data_stats['mean'], data_stats['std'])

    # Performing sliding window
    windowed_data = window_stack(all_data, stepsize=1, width=frame_size)

    # See how much data we have
    n_datapoints = windowed_data.shape[0]
    n_training_datapoints = int(n_datapoints*train_ratio)
    n_testing_datapoints = int(n_datapoints*test_ratio)

    train_data = all_data[:n_training_datapoints ]
    val_data = windowed_data[n_training_datapoints : - n_testing_datapoints]
    test_data = windowed_data[- n_testing_datapoints : ]

    # May have to perform some normalization, as they do?

    return train_data, val_data, test_data, data_stats


# pems7data = np.load("PEMS07.npz")["data"]
# pems7data = np.genfromtxt("PEMSD7_W_228.csv", delimiter = ",")
# ax3 = sns.heatmap(pems7data, cmap="YlGnBu")
# figure1 = ax3.get_figure()
# figure1.savefig('pems_1.png', dpi=400)


# TODOs:
#   Maybe do z-score normalization?
