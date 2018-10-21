import tensorflow as tf # Default graph is initialized when the library is imported
import os
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import scipy
from scipy import misc
# import matplotlib.pyplot as plt
# import cv2

# with tf.Graph().as_default() as graph: # Set default graph as graph

#            with tf.Session() as sess:
#                 # Load the graph in graph_def
#                 print("load graph")

#                 # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
#                 with gfile.FastGFile("model/trial4/frozen_model.pb",'rb') as f:

#                                 print("Load Image...")
#                                 # Read the image & get statstics
#                                 image = scipy.misc.imread('cifar/test/0_cat.png')
#                                 image = image.astype(float)
#                                 Input_image_shape=image.shape
#                                 height,width,channels = Input_image_shape

#                                 print("Plot image...")
#                                 #scipy.misc.imshow(image)

#                                 # Set FCN graph to the default graph
#                                 graph_def = tf.GraphDef()
#                                 graph_def.ParseFromString(f.read())
#                                 sess.graph.as_default()

#                                 # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

#                                 tf.import_graph_def(
#                                 graph_def,
#                                 input_map=None,
#                                 return_elements=None,
#                                 name="",
#                                 op_dict=None,
#                                 producer_op_list=None
#                                 )

#                                 # Print the name of operations in the session
#                                 for op in graph.get_operations():
#                                         print "Operation Name :",op.name         # Operation name
#                                         print "Tensor Stats :",str(op.values())     # Tensor name

#                                 # INFERENCE Here
#                                 l_input = graph.get_tensor_by_name('Inputs/fifo_queue_Dequeue:0') # Input Tensor
#                                 l_output = graph.get_tensor_by_name('upscore32/conv2d_transpose:0') # Output Tensor

#                                 print "Shape of input : ", tf.shape(l_input)
#                                 #initialize_all_variables
#                                 tf.global_variables_initializer()

#                                 # Run Kitty model on single image
#                                 Session_out = sess.run( l_output, feed_dict = {l_input : image} 


def main():
    checkpoint_directory = 'model/trial4/'
    checkpoint_file=tf.train.latest_checkpoint(checkpoint_directory)
    graph=tf.Graph()
    print("Load Image...")
    # Read the image & get statstics
    image = scipy.misc.imread('cifar/test/0_cat.png')
    image = image.astype(float)
    Input_image_shape=image.shape
    height,width,channels = Input_image_shape

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement =False)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess,checkpoint_file)
            input = graph.get_operation_by_name("main_parameters/Input").outputs[0]
            prediction=graph.get_operation_by_name("ArgMax").outputs[0]
            #newdata=put your data here
            print(sess.run(prediction,feed_dict={input:image}))


if __name__ == "__main__":
    main()