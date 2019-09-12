import numpy as np
import tensorflow as tf


def csv_to_npy(filename):
    quizzes = np.zeros((1000000, 81), np.int32)
    solutions = np.zeros((1000000, 81), np.int32)
    for i, line in enumerate(open(filename, 'r').read().splitlines()[1:]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s
    quizzes = quizzes.reshape((-1, 9, 9))
    np.save('quizzes', quizzes)

    solutions = solutions.reshape((-1, 9, 9))
    np.save('solutions', solutions)

def load_data():
    x = np.load('quizzes.npy')
    y = np.load('solutions.npy')
    return x[10:].astype(float), y[10:].astype(float), x[:10].astype(float), y[:10].astype(float)

def train_input_fn(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    dataset = dataset.shuffle(buffer_size=5).repeat().batch(batch_size)

    return dataset

