import numpy as np
import tensorflow as tf

from helper.data import get_data_set
from helper.model import model

from PIL import Image
# import numpy as np
import scipy
from scipy import misc


test_x, test_y = get_data_set("test")
x, y, output, y_pred_cls, global_step, learning_rate = model()


save_dir = 'model/trial4/'


saver = tf.train.Saver()
sess = tf.Session()

class_names = {'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,
               'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}

try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def infer(im):

	image = scipy.misc.imread(im)
	image = image.astype(float)
	image = np.array(image, dtype=float) / 255.0
	#image = image.reshape([-1, 3, 32, 32])
	#image = image.transpose([0, 2, 3, 1])
	image = image.reshape(-1, 32*32*3)

    # i = 0
	predicted_class = np.zeros(shape=len([image]), dtype=np.int)
	
	result = sess.run(y_pred_cls, feed_dict={x: image})
    # while i < len(test_x):
    #     j = min(i + _BATCH_SIZE, len(test_x))
    #     batch_xs = test_x[i:j, :]
    #     #batch_ys = test_y[i:j, :]
    #     predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs})
    #     i = j

    # correct = (np.argmax(test_y, axis=1) == predicted_class)
    # acc = correct.mean() * 100
    # correct_numbers = correct.sum()
	print(class_names[result[0]])
	# print('Cat -- should be 3!')

def main():
    infer('cifar/test/0_cat.png')
    infer('cifar/test/8_cat.png')
    infer('cifar/test/46_cat.png')
    infer('cifar/test/68_cat.png')
    infer('cifar/test/77_cat.png')
    infer('cifar/test/91_cat.png')
    infer('cifar/test/176_cat.png')
    

if __name__ == "__main__":
    main()


sess.close()