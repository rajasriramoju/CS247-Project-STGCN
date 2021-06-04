# Set up the model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input

class trainable_matrix(keras.layers.Layer):
    def __init__(self, shape):
        super(trainable_matrix, self).__init__()

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=shape, dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        # print("CALL\n\n")
        # print(inputs.shape)
        out = K.dot(inputs, self.w)
        return out



# Made of ST-conv blocks, which are temporal-spatial-temporal convolution layers
#   with GLU-relu-relu as the activations
#  There are two of these blocks, which have differnet number of channels in each
#   and then have dropout, though STGCN seems to keep all during training.
#  Then, there are two temporalconv layers with GLU, then sigmoid, then finally a 2d conv layer.
#  Remember that loss is from l2.
#  Each temporal 2dconv is 1 SAME + 1 VALID + activation
#  Each spatial 2dconv is 1 SAME + 1 gconv + activation
#   gconv does the following:
#    Takes the adjancy matrix (n_station, n_station) and dots with
#       reshaped input [batch_size*c_in, n_route] -> [batch_size*c_in, Ks*n_route]
#    Then it multiplies this output with a trainable kernel [Ks*c_in, c_out]
#      With reshaping, this is [batch_size*n_route, c_in*Ks] . [Ks*c_in, c_out] = [batch_size, n_route, c_out]
class STGCN():

  def __init__(self, channel_list, adj_matrix):

    self.channel_list = channel_list
    self.adj_matrix = adj_matrix

    self.conv_kernel_size = 3

    self.trainable_matrix = trainable_matrix((32,32))


  # BRO THIS TOOK SO LONG
  def create_model(self, input_shape):

    inputs = Input(shape=input_shape)
    channel_list = self.channel_list
    # First, iterate through our ST conv blocks
    # for i, channels in enumerate(self.channel_list):

    # ST conv block
    #  - Temporal Convolution
    #     - 2dconv, SAME (kernel size 1 x 1 x channels x new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length, n_stations, new_channels)
    #       - Note that this is separate from the following 2dconv, which operates on the original input to this layer.
    #       - Call this output x_in.    In addition, they slice from [3-1:seq_length] in order to match the next conv output.
    x_seq_length = tf.shape(inputs)[1]
    # print("SHOWED UP" + str(i))
    # print(x.shape)
    x_in = layers.Conv2D(channel_list[0][1], (1,1), padding="SAME")(inputs)

    x_in = x_in[:,self.conv_kernel_size-1:x_seq_length,:,:]

    #     - 2dconv, VALID (kernel size 3 x 1 x channels x 2*new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length-2, n_stations, 2*new_channels)
    #       - Call this output x_conv
    x_conv = layers.Conv2D(channel_list[0][1]*2, (self.conv_kernel_size,1), padding="VALID")(inputs)
    #     - Result (incorporates GLU):
    #       - x_conv[:,:,:,0:new_channels] + x_in + sigmoid(x_conv[:, :, :, -new_channels:])
    #       - (batch_size, seq_length-2, n_stations, new_channels)

    x_out1 = x_conv[:,:,:,0:channel_list[0][1] ] + x_in + tf.keras.activations.sigmoid(x_conv[:, :, :, -channel_list[0][1]:])
    #  - Spatial Convolution
    #     - 2dconv, SAME (kernel size 1 x 1 x channels x new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length, n_stations, new_channels)
    #       - Note that this is separate from the following 2dconv, which operates on the original input to this layer.
    #       - Call this output x_in.
    #       - IMPORTANT NOTE: This only happens if channels > new_channels, which is not true in STGCN.
    #       - So instead of the above behavior, x_in = original input
    x_in = x_out1

    #     - gconv
    #       - First, input is reshaped into: [batch_size*seq_len*channels, n_stations]

    _, seq_len, n, channel = x_out1.shape

    x_reshape = layers.Reshape((seq_len * channel, n))(x_out1)
    #       - input mat-multiplies with adj matrix, which gives [batch_size*seq_len*channel, n_stations]
    # According to the dot api, the axes correspond to the first and second input
    x_reshape = K.dot(x_reshape, K.variable(self.adj_matrix))
    #       - This then is reshaped into [batch_size*seq_len, channel, 1, n_stations]
    x_reshape = layers.Reshape((seq_len, channel, n), input_shape=(x_reshape.shape))(x_reshape)
    #       - This then is reshaped into [batch_size*seq_len*n_stations, channel]
    x_reshape = layers.Reshape((-1, channel), input_shape=(x_reshape.shape))(x_reshape)
    #       - THis is then matmuliplied with a trainable matrix (channel, channel)
    # x_gconv = K.dot(x_reshape, K.variable(self.trainable_matrix))
    x_gconv = self.trainable_matrix(x_reshape)

    #       - This results in [batch_size*seq_len*n_stations, channels], which is reshaped as
    #       - [batch_size*seq_len, n_stations, channel]
    #       - call this output x_gconv
    x_gconv = layers.Reshape((seq_len, n, channel), input_shape=(x_gconv.shape))(x_gconv)
    #     - Result (uses relu)
    #       - First, x_gconv is reshaped from [batch_size*seq_len, n_stations, channels] to
    #            - [batch_size,seq_len, n_stations, channels]
    #       - RELU ( x_gconv[:,:,:,0:new_channels] + x_in )
    #       - Same shape as output from previous temporal layer
    x_out2 = tf.keras.activations.relu( x_gconv[:,:,:,0:channel_list[0][1]] + x_in )


    #  - Temporal Convolution
    #     - 2dconv, SAME (kernel size 1 x 1 x channels x new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length, n_stations, new_channels)
    #       - Note that this is separate from the following 2dconv, which operates on the original input to this layer.
    #       - Call this output x_in.  In addition, they slice from [3-1:seq_length] in order to match the next conv output.
    x_seq_length = x_out2.shape[1]
    x_in = layers.Conv2D(channel_list[0][2], (1,1), padding="SAME")(x_out2)
    x_in = x_in[:,self.conv_kernel_size-1:x_seq_length,:,:]
    #     - 2dconv, VALID (kernel size 3 x 1 x channels x new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length-2, n_stations, new_channels)
    #       - Call this output x_conv
    x_conv = layers.Conv2D(channel_list[0][2], (self.conv_kernel_size,1), padding="VALID")(x_out2)
    #     - Result with ReLU
    #       - relu of (x_conv + x_in)
    #       - (batch_size, seq_length-2, n_stations, new_channels)
    x = tf.keras.activations.relu( x_conv + x_in)

    # ST conv block
    #  - Temporal Convolution
    #     - 2dconv, SAME (kernel size 1 x 1 x channels x new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length, n_stations, new_channels)
    #       - Note that this is separate from the following 2dconv, which operates on the original input to this layer.
    #       - Call this output x_in.    In addition, they slice from [3-1:seq_length] in order to match the next conv output.
    x_seq_length = x.shape[1]
    # print("SHOWED UP" + str(i))
    # print(x.shape)
    x_in = layers.Conv2D(channel_list[1][1], (1,1), padding="SAME")(x)
    x_in = x_in[:,self.conv_kernel_size-1:x_seq_length,:,:]
    #     - 2dconv, VALID (kernel size 3 x 1 x channels x 2*new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length-2, n_stations, 2*new_channels)
    #       - Call this output x_conv
    x_conv = layers.Conv2D(channel_list[1][1]*2, (self.conv_kernel_size,1), padding="VALID")(x)
    #     - Result (incorporates GLU):
    #       - x_conv[:,:,:,0:new_channels] + x_in + sigmoid(x_conv[:, :, :, -new_channels:])
    #       - (batch_size, seq_length-2, n_stations, new_channels)
    x_out1 = x_conv[:,:,:,0:channel_list[1][1] ] + x_in + tf.keras.activations.sigmoid(x_conv[:, :, :, -channel_list[1][1]:])
    #  - Spatial Convolution
    #     - 2dconv, SAME (kernel size 1 x 1 x channels x new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length, n_stations, new_channels)
    #       - Note that this is separate from the following 2dconv, which operates on the original input to this layer.
    #       - Call this output x_in.
    #       - IMPORTANT NOTE: This only happens if channels > new_channels, which is not true in STGCN.
    #       - So instead of the above behavior, x_in = original input
    x_in = x_out1



    #     - gconv
    #       - First, input is reshaped into: [batch_size*seq_len*channels, n_stations]
    _, seq_len, n, channel = x_out1.shape
    x_reshape = layers.Reshape((seq_len * channel, n))(x_out1)
    #       - input mat-multiplies with adj matrix, which gives [batch_size*seq_len*channel, n_stations]
    x_reshape = K.dot(x_reshape, K.variable(self.adj_matrix))
    #       - This then is reshaped into [batch_size*seq_len, channel, 1, n_stations]
    x_reshape = layers.Reshape((-1, seq_len, channel, n), input_shape=(x_reshape.shape))(x_reshape)
    #       - This then is reshaped into [batch_size*seq_len*n_stations, channel]
    x_reshape = layers.Reshape((-1, channel), input_shape=(x_reshape.shape))(x_reshape)
    #       - THis is then matmuliplied with a trainable matrix (channel, channel)
    # x_gconv = K.dot(x_reshape, K.variable(self.trainable_matrix))
    x_gconv = self.trainable_matrix(x_reshape)
    #       - This results in [batch_size*seq_len*n_stations, channels], which is reshaped as
    #       - [batch_size*seq_len, n_stations, channel]
    #       - call this output x_gconv
    x_gconv = layers.Reshape((seq_len, n, channel), input_shape=(x_gconv.shape))(x_gconv)
    #     - Result (uses relu)
    #       - First, x_gconv is reshaped from [batch_size*seq_len, n_stations, channels] to
    #            - [batch_size,seq_len, n_stations, channels]
    #       - RELU ( x_gconv[:,:,:,0:new_channels] + x_in )
    #       - Same shape as output from previous temporal layer
    x_out2 = tf.keras.activations.relu( x_gconv[:,:,:,0:channel_list[1][1]] + x_in )

    #  - Temporal Convolution
    #     - 2dconv, SAME (kernel size 1 x 1 x channels x new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length, n_stations, new_channels)
    #       - Note that this is separate from the following 2dconv, which operates on the original input to this layer.
    #       - Call this output x_in.  In addition, they slice from [3-1:seq_length] in order to match the next conv output.
    x_seq_length = x_out2.shape[1]
    x_in = layers.Conv2D(channel_list[1][2], (1,1), padding="SAME")(x_out2)
    x_in = x_in[:,self.conv_kernel_size-1:x_seq_length,:,:]
    #     - 2dconv, VALID (kernel size 3 x 1 x channels x new_channels)
    #       - (batch_size, seq_length, n_stations, channels) -> (batch_size, seq_length-2, n_stations, new_channels)
    #       - Call this output x_conv
    x_conv = layers.Conv2D(channel_list[1][2], (self.conv_kernel_size,1), padding="VALID")(x_out2)
    #     - Result with ReLU
    #       - relu of (x_conv + x_in)
    #       - (batch_size, seq_length-2, n_stations, new_channels)
    x = tf.keras.activations.relu( x_conv + x_in)

    # Now we move on to the output layers

    # First, we do dropout
    x = layers.Dropout(0.0)(x) # Set dropout to whatever you want

    _, T, n, channel = x.get_shape().as_list()
    # Run Temporal conv with GLU
    x_seq_length = x.shape[1]
    x_in = layers.Conv2D(channel, (1,1), padding="SAME")(x)
    x_in = x_in[:,T-1:x_seq_length,:,:]
    x_conv = layers.Conv2D(channel*2, (T,1), padding="VALID")(x)
    x_out_final1 = x_conv[:,:,:,0:channel ] + x_in + tf.keras.activations.sigmoid(x_conv[:, :, :, -channel:])

    # Run Temporal conv with sigmoid (no need for x_in anymore)
    x_conv = layers.Conv2D(channel, (1,1), padding="VALID")(x_out_final1)
    x_out_final2 = tf.keras.activations.sigmoid(x_conv)

    # A final 2d conv with filter (1,1,channel,1)
    x_out_final3 = layers.Conv2D(1, (1,1), padding="VALID")(x_out_final2)

    # Create model
    model = Model(inputs, x_out_final3)

    return model

# The output of the model is actually (None, 1, 228, 1)
#  So during testing, they have to do this one step at a time (e.g. to predict 9 steps
#    they repeat this process 9 times.)

# Model Output shapes for reference:
# Channels - a parameter
# [1, 32, 64]
# STBlock Input
# (None, 12, 228, 1)
# Temporal conv1 output
# (None, 10, 228, 32)
# Spatio conv1 output
# (None, 10, 228, 32)
# Temporal conv2 output
# (None, 8, 228, 64)

# Channels - a parameter
# [64, 32, 128]
# STBlock Input
# (None, 8, 228, 64)
# Temporal conv1 output
# (None, 6, 228, 32)
# Spatio conv1 output
# (None, 6, 228, 32)
# Temporal conv2 output
# (None, 4, 228, 128)

# Blocks Finished, input to out layer
# (None, 4, 228, 128)
# Temporal conv1
# (None, 1, 228, 128)
# Temporal conv1
# (None, 1, 228, 128)
# "FC" layer, actually a 2d conv
# (None, 1, 228, 1)
# Output Result
# (None, 1, 228, 1)



# TODOS:
#  - Layer norm
