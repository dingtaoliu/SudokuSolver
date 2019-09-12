import tensorflow as tf 
import data
import model
import numpy as np

if __name__ == "__main__":
    sudoku_solver = tf.estimator.Estimator(
        model_fn=model.cnn_model_fn, model_dir="./model_output"
    )

    train_x, train_y, test_x, test_y = data.load_data()

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=64,
        num_epochs=None,
        shuffle=True
    )

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        y=test_y,
        shuffle=False
    )

    print("\n\n\nTraining\n\n\n")
    sudoku_solver.train(
        input_fn=train_input_fn,
        steps=5000
    )

    print("\n\n\nTesting\n\n\n")
    print(test_x.shape)
    print(test_y.shape)
    predictions = sudoku_solver.predict(input_fn=test_input_fn)

    # template = ("\nPrediction: \n {} \nExpected: \n {} \n")

    for p, solution in zip(predictions, test_y):
        guess = np.round(p["solutions"])
        print(guess)
        print(solution)


