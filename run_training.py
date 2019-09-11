import tensorflow as tf 
import data
import model 

if __name__ == "__main__":
    sudoku_solver = tf.estimator.Estimator(
        model_fn=model.cnn_model_fn, model_dir="/tmp/cnn_model"
    )

    x, y = data.load_data()

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": x},
    #     y=y,
    #     batch_size=64,
    #     num_epochs=None,
    #     shuffle=True
    # )

    sudoku_solver.train(
        input_fn=lambda:data.train_input_fn(x, y, 64),
        steps=1
    )

