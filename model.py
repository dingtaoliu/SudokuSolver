import tensorflow as tf 
import os
import sklearn.model_selection as ms
import numpy as np

num_incep_modules = 2
checkpoint_path = "/home/danny/Documents/repos/SudokuSolver/model_checkpoints"

def simple_conv_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(81 * 9),
        tf.keras.layers.Reshape((-1, 9)),
        tf.keras.layers.Activation('softmax')
    ])

    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def inception_model():
    input_layer = tf.keras.Input(shape=(9, 9, 1))
    h = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(input_layer)
    #h = tf.keras.layers.MaxPool2D(32, (5, 5), padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(h)
    h = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    #h = tf.keras.layers.MaxPool2D(32, (3, 3), padding='same')(h)

    for _ in range(num_incep_modules):
        # 1 x 1 filter
        h1 =  tf.keras.layers.Conv2D(64, (1, 1), padding='same')(h)
        h1 = tf.keras.layers.BatchNormalization()(h1)

        # 3 x 3 filter
        h3 = tf.keras.layers.Conv2D(32, (1, 1), padding='same')(h)
        h3 = tf.keras.layers.Conv2D(64, (3, 1), padding='same')(h3)
        h3 = tf.keras.layers.Conv2D(64, (1, 3), padding='same')(h3)
        h3 = tf.keras.layers.BatchNormalization()(h3)

        # 5 x 5 filter
        h5 = tf.keras.layers.Conv2D(16, (1, 1), padding='same')(h)
        h5 = tf.keras.layers.Conv2D(32, (9, 1), padding='same')(h5)
        h5 = tf.keras.layers.Conv2D(32, (1, 9), padding='same')(h5)
        h5 = tf.keras.layers.BatchNormalization()(h5)

        h = tf.keras.layers.Concatenate()([h, h1, h3, h5])

    h = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same')(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(81 * 9)(h)
    h = tf.keras.layers.Reshape((-1, 9))(h)
    output_layer = tf.keras.layers.Activation('softmax')(h)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def full_inception_model():
    input_layer = tf.keras.Input(shape=(9, 9, 1))
    h = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(input_layer)
    #h = tf.keras.layers.MaxPool2D(32, (3, 3), padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    for _ in range(num_incep_modules):
        # 1 x 1 filter
        h1 =  tf.keras.layers.Conv2D(64, (1, 1), padding='same')(h)
        h1 = tf.keras.layers.BatchNormalization()(h1)
        h1 = tf.keras.layers.ReLU()(h1)

        # 3 x 3 filter
        h3 = tf.keras.layers.Conv2D(32, (1, 1), padding='same')(h)
        h3 = tf.keras.layers.Conv2D(64, (3, 1), padding='same')(h3)
        h3 = tf.keras.layers.Conv2D(64, (1, 3), padding='same')(h3)
        h3 = tf.keras.layers.BatchNormalization()(h3)
        h3 = tf.keras.layers.ReLU()(h3)

        # 5 x 5 filter
        h5 = tf.keras.layers.Conv2D(16, (1, 1), padding='same')(h)
        h5 = tf.keras.layers.Conv2D(32, (9, 1), padding='same')(h5)
        h5 = tf.keras.layers.Conv2D(32, (1, 9), padding='same')(h5)
        h5 = tf.keras.layers.BatchNormalization()(h5)
        h5 = tf.keras.layers.ReLU()(h5)

        h = tf.keras.layers.Concatenate()([input_layer, h, h1, h3, h5])
        #h = tf.keras.layers.Dropout(0.5)(h)

    h = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same')(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(81 * 9)(h)
    h = tf.keras.layers.Reshape((-1, 9))(h)
    output_layer = tf.keras.layers.Activation('softmax')(h)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


x = np.load('quizzes.npy').reshape(-1, 9, 9, 1)
y = np.load('solutions.npy').reshape(-1, 81, 1) - 1

x2 = np.load('solutions.npy').reshape(-1)
indices = np.random.choice(np.arange(x2.size), replace=False, size=int(x2.size * 0.2))
add = indices[:indices.size // 2]
sub = indices[indices.size // 2:]
x2[add] += 1
x2[sub] -= 1
x2 = x2.reshape(-1, 9, 9, 1)

x = np.concatenate((x, x2), axis=0)
y = np.concatenate((y, y), axis=0)

# x3 = np.load('solutions.npy').reshape(-1)
# indices = np.random.choice(np.arange(x3.size), replace=False, size=int(x3.size * 0.5))
# add = indices[:indices.size // 2]
# sub = indices[indices.size // 2:]
# x3[add] += 1
# x3[sub] -= 1
# x3 = x2.reshape(-1, 9, 9, 1)

# x_step2 = np.concatenate((x, x3), axis=0)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.1, random_state = 5)

    model = full_inception_model()
    print(X_train.shape, y_train.shape)

    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(X_train, y_train, epochs=1, batch_size=256, callbacks=[cp_callback])
    model.evaluate(X_test, y_test, verbose=2)

    #X_train, X_test, y_train, y_test = ms.train_test_split(x_step2, y, test_size=0.1, random_state = 5)

    #model.save("incep_model.h5")
