import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size, frame_size):
        'Initialization'
        self.data = data
        self.num_datapoints = data.shape[0]
        self.batch_size = batch_size
        self.frame_size = frame_size # number of datapoints/features which constitute both historical traffic data and predictions
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_datapoints / self.batch_size))

    def __getitem__(self, index):
        # Generate X random indexes in self.data
        start_indexes = np.random.choice(np.arange(self.num_datapoints - self.frame_size), self.batch_size)

        X = []
        y = []
        # Find the historical data and data to predict for each start index
        for idx in start_indexes:
            x_item = self.data[idx:idx+self.frame_size-1]
            y_item = np.expand_dims(self.data[idx+self.frame_size-1], 0)

            # Expand dims more
            x_item = np.expand_dims(x_item, -1)
            y_item = np.expand_dims(y_item, -1)
            X.append( x_item )
            y.append( y_item )
            # Should be of shape (batch_size, seq_length, num_stations, 1)

        X = np.array(X)
        y = np.array(y)

        return X, y

    def generate_data(self):
        'Generate one batch of data'

        while True:

            # Generate X random indexes in self.data
            start_indexes = np.random.choice(np.arange(self.num_datapoints - self.frame_size), self.batch_size)

            X = []
            y = []
            # Find the historical data and data to predict for each start index
            for idx in start_indexes:
                x_item = self.data[idx:idx+self.frame_size-1]
                y_item = np.expand_dims(self.data[idx+self.frame_size-1], 0)

                # Expand dims more
                x_item = np.expand_dims(x_item, -1)
                y_item = np.expand_dims(y_item, -1)
                X.append( x_item )
                y.append( y_item )
                # Should be of shape (batch_size, seq_length, num_stations, 1)

            X = np.array(X)
            yield (X, y)

    def on_epoch_end(self):  # Shuffles our data
        'Updates indexes after each epoch'
        indexes = np.arange(len(self.data))
        np.random.shuffle(indexes)
        self.data = self.data[indexes]
