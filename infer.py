import numpy as np
import tensorflow as tf

from helper.data import get_data_set
from helper.model import model

from PIL import Image
# import numpy as np
import scipy
from scipy import misc

import matplotlib.pyplot as plt


test_x, test_y = get_data_set("test")
x, y, output, y_pred_cls, global_step, learning_rate = model()


save_dir = 'model/trial4/'


saver = tf.train.Saver()
sess = tf.Session()

try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def infer(im):
    saver = tf.train.Saver()
    sess = tf.Session()

    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(sess, save_path=last_chk_path)

    image = scipy.misc.imread(im)
    image = image.astype(float)
    image = np.array(image, dtype=float) / 255.0
    #image = image.reshape([-1, 3, 32, 32])
    #image = image.transpose([0, 2, 3, 1])
    image = image.reshape(-1, 32*32*3)

    # i = 0
    predicted_class = np.zeros(shape=len([image]), dtype=np.int)
    
    result = sess.run(y_pred_cls, feed_dict={x: image})

    class_names = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

    print('-'*50)
    print('The {} is classified as:'.format(im))
    print(class_names[result[0]])
    print('-'*50)
    # print('Cat -- should be 3!')


def get_activations(im, var_name = "conv1_layer/conv2d/Conv2D"):
    saver = tf.train.Saver()
    sess = tf.Session()

    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(sess, save_path=last_chk_path)

    image = scipy.misc.imread(im)
    image = image.astype(float)
    image = np.array(image, dtype=float) / 255.0
    #image = image.reshape([-1, 3, 32, 32])
    #image = image.transpose([0, 2, 3, 1])
    image = image.reshape(-1, 32*32*3)

    result = sess.run('conv1_layer/conv2d/Conv2D:0' ,feed_dict={x:[image.reshape([3072])]})  

    # output_cl1 = restore_see_layer(ix=image,model_name=base_model,var_name='conv1_layer/conv2d/Conv2D')
    # print(result.shape)

    # print(result[0][0].shape)

    filters = 32
    fig = plt.figure()
    plt.figure(1, figsize=(50,50))
    n_columns = 6
    n_rows = 6
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        # plt.title('Filter ' + str(i), fontsize=10)
        plt.imshow(result[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.savefig('CONV_rslt.png')


sess.close()