import numpy as np
from utils import weight_computation, create_train_val_test, evalCallback, eval_next_steps
from model_builder import STGCN
from data_generator import DataGenerator
import tensorflow as tf

# Load in the preprocessed data - replace with your own data paths
distance_matrix = np.genfromtxt("preprocessed/W_228_new.csv", delimiter = ",")
# Make sure our distance matrix is converted to weights according to computation
weight_matrix = weight_computation(distance_matrix)

dataset = np.genfromtxt("preprocessed/V_228_new.csv", delimiter = ",")

# Create train test set
n_history = 12 # Num datapoints in feature vector
n_pred = 9     # Num datapoints to predict later in time from feature vector
split_config = {"train": 0.6, "val": 0.2, "test": 0.2}

# Note that train data is slightly different than the val and test_data:
#  train_data is a full sequence, which we sample (n_history + n_pred datapoints)
#      to predict a next value.
#  val and test are 'frames', where each datapoint consists of n_history + n_pred datapoints.
train_data, val_data, test_data, data_stats = create_train_val_test(dataset, n_history + n_pred, split_config)

# Create our model
channel_list = [[1, 32, 64], [64, 32, 128]]
batch_size = 64
seq_length = 12
num_stations = 228
input_mat = np.zeros((batch_size, seq_length, num_stations, 1))
epochs = 3

# Initialize model
model_init = STGCN(channel_list, weight_matrix)

# Prepare data generators and training
train_generator = DataGenerator(train_data, batch_size, n_history+1)

x,y = train_generator.__getitem__(0)
model = model_init.create_model(x[0].shape)
# Compile model with rmsprop and mse
model.compile(optimizer="rmsprop", loss= tf.keras.losses.MeanSquaredError() )

# Tell me how large our data is
print("DATA SHAPES:\n")
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

# Train our model with the generators
model.fit(train_generator, \
    steps_per_epoch= train_data.shape[0] // batch_size, \
    epochs= epochs, callbacks=[evalCallback(val_data, n_history, n_pred, model, data_stats)])


# Predict on the test set
print("\n\nChecking performance on Test Data")
eval_next_steps(model, test_data, n_history, n_pred, data_stats)
