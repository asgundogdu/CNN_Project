import tensorflow as tf


def model():
    image_size = 32
    num_channels = 3
    num_classes = 10


    # Using name scope to use/understand tensorboard while debugging

    with tf.name_scope('main_parameters'):
        x = tf.placeholder(tf.float32, shape=[None, image_size * image_size * num_channels], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, num_classes], name='Output')
        x_image = tf.reshape(x, [-1, image_size, image_size, num_channels], name='images')

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1_layer') as scope:
        conv = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[5, 5],
            strides=(1, 1),
            padding='SAME',
            activation=tf.nn.relu
        )
        # conv = tf.layers.conv2d(
        #     inputs=conv,
        #     filters=64,
        #     kernel_size=[4, 4],
        #     strides=(1, 1),
        #     padding='SAME',
        #     activation=tf.nn.relu
        # )
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.3, name=scope.name)

    with tf.variable_scope('conv2_layer') as scope:
        conv = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=256,
            kernel_size=[2, 2],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.3, name=scope.name)

    with tf.variable_scope('fully_connected_layer') as scope:
        flat = tf.reshape(drop, [-1, 4 * 4 * 256])

        fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
        drop = tf.layers.dropout(fc, rate=0.6)

        fc = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
        drop = tf.layers.dropout(fc, rate=0.4)

        softmax = tf.layers.dense(inputs=drop, units=num_classes, activation=tf.nn.softmax, name=scope.name)

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, global_step, learning_rate


def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate