import tensorflow as tf 
import os
import sklearn.model_selection as ms
import numpy as np

from model import *


def load_model():
    model = full_inception_model()
    weights = tf.train.latest_checkpoint(".")
    model.load_weights(weights)
    return model

X = np.load('quizzes.npy').reshape(-1, 9, 9, 1)
Y = np.load('solutions.npy').reshape(-1, 9, 9)

def test(num, shuffle=False):
    test_x = X[:num].reshape(-1, 9, 9, 1)
    pred = model.predict(test_x)
    pred = np.argmax(pred, axis=2).reshape(-1, 9, 9) + 1
    
    for i in range(num):
        diff = pred[i] - Y[i]
        incorrect = np.where(diff != 0).size 

        if incorrect != 0:
            print("Prediction:\n")
            print(pred[i])
            print("Actual:\n")
            print(Y[i])


def iterative_test(num, shuffle=False):
    test_x = X[:num].reshape(-1, 9, 9, 1)
    pred = model.predict(test_x)
    pred = np.argmax(pred, axis=2).reshape(-1, 9, 9) + 1
    
    for i in range(num):
        diff = pred[i] - Y[i]
        incorrect = np.where(diff != 0).size 

        if incorrect != 0:
            print("Prediction:\n")
            print(pred[i])
            print("Actual:\n")
            print(Y[i])

if __name__ == "__main__":
    model = load_model()
    test(5)

