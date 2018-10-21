# Import Packages
import numpy as np
import tensorflow as tf
from time import time
import math

# Import built-in functions
from helper.data import get_data_set
from helper.model import model, lr

train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
tf.set_random_seed(123)
x, y, output, y_pred_cls, global_step, learning_rate = model()
global_accuracy = 0
epoch_start = 0

# Parametes
batch_size_ = 128
eopch_num = 80
save_dir = "./model/trial4/"

# Loss function and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9, beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)

# Evaluation Metrics
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# For Saving the Model
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(save_dir, sess.graph)

try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())

def train(epoch):
    global epoch_start
    global global_accuracy
    epoch_start = time()
    batch_size = int(math.ceil(len(train_x) / batch_size_))
    i_global = 0

    for batch in range(batch_size):
        batch_xs = train_x[batch * batch_size_: (batch + 1) * batch_size_]
        batch_ys = train_y[batch * batch_size_: (batch + 1) * batch_size_]

        start_time = time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)})
        duration = time() - start_time


        # if s % 10 == 0:
        #     percentage = int(round((s/batch_size)*100))

        #     bar_len = 29
        #     filled_len = int((bar_len*int(percentage))/100)
        #     bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)


        #     msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
        #     print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, batch_size_ / duration))

    # print('#'*120)

	# Compute Training Accuracy
    i = 0
    predicted_class = np.zeros(shape=len(train_x), dtype=np.int)
    while i < len(train_x):
        j = min(i + batch_size_, len(train_x))
        batch_xs = train_x[i:j, :]
        batch_ys = train_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)}
        )
        i = j

    correct = (np.argmax(train_y, axis=1) == predicted_class)
    train_acc = correct.mean()*100  

	# Compute Test Accuracy
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + batch_size_, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    test_acc = correct.mean()*100    

    msg = "Epoch {} - Training Accuracy: {:.4f} - Test Accuracy: {:.4f}"
    print(msg.format((epoch+1), train_acc, test_acc))
    

    #########################################################################################

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + batch_size_, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Test accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, i_global)

        saver.save(sess, save_path=save_dir, global_step=i_global)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("-"*120)


def freeze_graph(model_dir, output_node_names):
	if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

     # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def main():
    train_start = time()

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    #parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    #args = parser.parse_args()

    for i in range(eopch_num):
        print("\nEpoch: {}/{}\n".format((i+1), eopch_num))
        train(i)

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(global_accuracy, int(hours), int(minutes), seconds))

    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    freeze_graph("./model/trial4/", "fully_connected_layer/softmax")


if __name__ == "__main__":
    main()


sess.close()