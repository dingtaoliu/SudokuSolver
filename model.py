import tensorflow as tf 

from data import *
import sklearn.model_selection as ms

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(81 * 9),
    tf.keras.layers.Reshape((-1, 9)),
    tf.keras.layers.Activation('softmax')
])



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

x = np.load('quizzes.npy').reshape(-1, 9, 9, 1)
y = np.load('solutions.npy').reshape(-1, 81, 1)

y = y - 1

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.1, random_state = 5)

    print(X_train.shape, y_train.shape)

    model.fit(X_train, y_train, epochs=1)
    model.evaluate(X_test, y_test, verbose=2)
