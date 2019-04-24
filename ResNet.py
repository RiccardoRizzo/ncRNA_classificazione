from keras.layers import Conv2D, \
                         BatchNormalization,\
                         Activation, \
                         Input, \
                         GlobalAveragePooling2D,\
                         Dropout, \
                         Dense

from keras.layers import Conv1D, GlobalAveragePooling1D

from keras.models import Sequential, Model
# from keras.regularizers import l2
from keras import optimizers

from keras import layers
from keras import regularizers
from keras import models

from keras import backend as K

from keras.activations import relu, softmax
from keras.layers.merge import add

from keras.utils.vis_utils import plot_model

import yaml

#===============================================================================
# Rete con
# <input_sh> ingressi e
# <dl2_units_num> uscite
# la rete con piu' uscite trasforma l'uscita in categorie
# e pretende una attivazione softmax
#===============================================================================


#===============================================================================
# BLOCCO ELEMENTARE RESNET
def block(n_output, upscale=False):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):

        # H_l(x):
        # first pre-activation
        h = layers.BatchNormalization()(x)
        h = layers.Activation(relu)(h)
        # first convolution
        h = layers.Conv1D(kernel_size=3, filters=n_output, strides=1,
            padding='same', kernel_regularizer=regularizers.l2(0.01))(h)

        # second pre-activation
        h = layers.BatchNormalization()(x)
        h = layers.Activation(relu)(h)
        # second convolution
        h = layers.Conv1D(kernel_size=3, filters=n_output, strides=1,
            padding='same', kernel_regularizer=regularizers.l2(0.01))(h)

        # f(x):
        if upscale:
            # 1x1 conv2d
            f = layers.Conv1D(kernel_size=1, filters=n_output,
                strides=1, padding='same')(x)
        else:
            # identity
            f = x

        # F_l(x) = f(x) + H_l(x):
        return add([f, h])

    return f
#===============================================================================
def Net_f(filePar):
    # legge il file di PARAMETRI
    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        # CARICO I PARAMETRI DELLA RETE
        input_shape = cfg["parAlg"]["input_shape"]
        input_dim = input_shape[0]
        input_channels = input_shape[1]

        dropout = cfg["parAlg"]["dropout"]

        num_ker = cfg["parAlg"]["num_ker"]
        dim_ker = cfg["parAlg"]["dim_ker"]

        dim_dense = cfg["parAlg"]["dim_dense"]

        learning_rate = cfg["parAlg"]["learning_rate"]
        regularizer = cfg["parAlg"]["regularizer"]

    # dimensiona e restituisce la rete
    return Net_ResNet(input_dim, input_channels, \
                    num_ker, dim_ker, \
                    dim_dense, \
                    dropout, regularizer, learning_rate)


#===============================================================================
def Net_ResNet(input_dim, input_channels, \
                num_ker, dim_ker,
                dim_dense, \
                dropout, regularizer, learning_rate):

    # il default di Keras e' image height, image width, image channels, quindi va bene 28, 28, 1
    #img_height, img_width, img_channels = 28, 28 , 1

    image_tensor = layers.Input(shape=(input_dim, input_channels))

    # first conv2d with post-activation to transform the input data to some reasonable form
    x = layers.Conv1D(kernel_size=dim_ker, filters=num_ker, strides=1, padding='same',
        kernel_regularizer=regularizers.l2(regularizer))(image_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)

    # F_1
    x = block(num_ker)(x)
    # F_2
    x = block(num_ker)(x)
    # F_2
    x = block(num_ker)(x)


    # F_3
    # H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
    # and we can't add together tensors of inconsistent sizes, so we use upscale=True
    # x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
    # F_4
    # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation
    # F_5
    # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

    # F_6
    # x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
    # F_7
    # x = block(48)(x)                     # !!! <------- Uncomment for local evaluation

    # last activation of the entire network's output
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)

    # average pooling across the channels
    # 28x28x48 -> 1x48
    x = layers.GlobalAveragePooling1D()(x)

    # dropout for more robust learning
    x = layers.Dropout(dropout)(x)

    # last softmax layer
    x = layers.Dense(units=dim_dense, kernel_regularizer=regularizers.l2(regularizer))(x)
    x = layers.Activation(softmax)(x)

    model = models.Model(inputs=image_tensor, outputs=x)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    optim = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy')

    return model
#-------------------------------------------------------------------------------
