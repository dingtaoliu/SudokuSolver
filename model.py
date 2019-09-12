import tensorflow as tf 

def cnn_model_fn(features, labels, mode):


    input_layer = tf.reshape(features["x"], [-1, 9, 9, 1])
    


    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    # pool1 = tf.layers.max_pooling2d(
    #     inputs=conv1, 
    #     pool_size=[2, 2], 
    #     strides=1
    # )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    output = tf.layers.conv2d(
        inputs=conv2,
        filters=1,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu
    )

    predictions = {
        "solutions": output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    labels = tf.reshape(labels, [-1, 9, 9, 1])
    print("######################################")
    print(output.shape)
    print(labels.shape)
    print("#######################################")

    loss = tf.losses.absolute_difference(labels, output)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["solutions"]
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )
