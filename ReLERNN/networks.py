'''
Authors: Jeff Adrion, Andrew Kern, Jared Galloway
Modified by J.Min
'''

from ReLERNN.imports import *

class ShuffleIndividuals(keras.layers.Layer):
    """
    Randomly permutes individuals
    Moved from data batch generation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
         batch_size = tf.shape(inputs)[0]
         n_inds = tf.shape(inputs)[2]
         shuffled_idx = tf.argsort(tf.random.uniform([batch_size, n_inds]), axis=1)
         inputs = tf.transpose(inputs, perm=[0, 2, 1])
         inputs = tf.gather(inputs, shuffled_idx, axis=1, batch_dims=1)
         inputs = tf.transpose(inputs, perm=[0, 2, 1])
         return inputs

def GRU_TUNED84(x,y):
    '''
    Same as GRU_VANILLA but with dropout AFTER each dense layer.
    First layer shuffles individual
    '''

    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = layers.Input(shape=(numSNPs,numSamps))

    # additional layer - shuffling individuals
    model = ShuffleIndividuals()(genotype_inputs)
    model = layers.Bidirectional(layers.GRU(84,return_sequences=False))(model)
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)

    #----------------------------------------------------

    position_inputs = layers.Input(shape=(numPos,))
    m2 = layers.Dense(256)(position_inputs)

    #----------------------------------------------------


    model =  layers.concatenate([model,m2])
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    #----------------------------------------------------

    model = Model(inputs=[genotype_inputs,position_inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse', jit_compile=False)
    model.summary()

    return model


def GRU_POOLED(x,y):

    sites=x.shape[1]
    features=x.shape[2]

    genotype_inputs = layers.Input(shape=(sites,features))
    model = layers.Bidirectional(layers.GRU(84,return_sequences=False))(genotype_inputs)
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    model = Model(inputs=[genotype_inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse', jit_compile=False)
    model.summary()

    return model


def HOTSPOT_CLASSIFY(x,y):

    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = layers.Input(shape=(numSNPs,numSamps))
    # additional layer - shuffle individuals
    model = ShuffleIndividuals()(genotype_inputs)
    model = layers.Bidirectional(layers.GRU(84,return_sequences=False))(model)
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)

    #----------------------------------------------------

    position_inputs = layers.Input(shape=(numPos,))
    m2 = layers.Dense(256)(position_inputs)

    #----------------------------------------------------


    model =  layers.concatenate([model,m2])
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1,activation='sigmoid')(model)

    #----------------------------------------------------

    model = Model(inputs=[genotype_inputs,position_inputs], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', jit_compile=False)
    model.summary()

    return model
