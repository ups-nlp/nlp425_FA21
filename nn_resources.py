#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:01:07 2021

@author: prowe
"""

# Installed modules
import tensorflow as tf

# My modules
import config



def model(inputs, Ws, bs):
    """
    The neural network model: Cast the inputs onto a tensor (tensorflow
    requires this), get the weighted sum of the the inputs and then pass it
    through the activation function (a sigmoid).

    @inputs The vectors representing the phrases
    @return Weighted sum of inputs after passing through activation function

    Missing from the inputs, but within the scope (yuck):
    weights:list, intercepts:list):
    @weights Ws The weights for each
    @intercepts bs The intercepts for each

    """
    tensors = tf.cast(inputs, tf.float32)
    #for i in range(len(Ws)):
    #    result = tf.sigmoid(tf.matmul(tensors, Ws[i]) + bs[i])
    result = [tf.sigmoid(tf.matmul(tensors, Ws[i]) + bs[i]) \
              for i in range(len(Ws))]
    return tf.squeeze(result)




def run_step(inputs, labels, opt, Ws, bs):
    """ Run a training step
    @param inputs
    @param labels The correct answers
    @param opt The optimizer
    @param Ws The weights
    @param bs The constants
    """
    with tf.GradientTape() as tape:
        prediction = model(inputs, Ws, bs)
        loss = tf.keras.losses.mae(labels, prediction)
        print('loss:')
        tf.print(loss)
        # loss = np.mean(np.abs(labels - prediction), axis=0)

    trainable_vars = Ws + bs # [W for W in Ws] + [b for b in bs]

    # These lines were not tabbed over. Why not?
    # Calculate gradients
    gradients = tape.gradient(loss, trainable_vars)

    # Change the model (w and b) based on gradient
    opt.apply_gradients(zip(gradients, trainable_vars))

    if config.VERBOSITY:
        print('labels'); tf.print(labels)
        print('prediction'); tf.print(prediction)
        print('loss:'); tf.print(loss)
        print('gradients:'); tf.print(gradients)
        print('trainable_vars:', trainable_vars)
        #print('weights:'); tf.print(Ws)
        #print('intercept:'); tf.print(bs)





def train_model(x, y, EPOCHS, BATCH_SIZE, INIT_LR, Ws, bs):
    """
    Train the model
    @param x The features
    @param y The labels (correct answers)
    @param EPOCHS
    @param BATCH_SIZE
    @param INIT_LR

    @return loss
    @return prediction
    """

    # The optimizer contains the model w and b that will be optimized
    # through back propagation * I assume *
    opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)

    num_updates  = int(x.shape[0] / BATCH_SIZE)

    # training
    for epoch in range(EPOCHS):
        print()
        print(f'starting epoch {epoch} / {EPOCHS}')
        if config.VERBOSITY:
            print(f'y: {y[:BATCH_SIZE]}')

        for i in range(num_updates):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE

            run_step(x[start:end], y[start:end], opt, Ws, bs)


    # testing
    prediction = model(x, Ws, bs)
    loss = tf.keras.losses.mse(y, prediction)
    print('final loss')
    tf.print(loss)

    return loss, prediction
