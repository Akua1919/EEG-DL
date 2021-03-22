import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

model_file=tf.train.latest_checkpoint('./First_Try_Model/Model_Saver/')
saver = tf.train.import_meta_graph(model_file+'.meta')

sess = tf.InteractiveSession()
saver.restore(sess, model_file)
x = tf.get_default_graph().get_tensor_by_name("Convolutional_1/W_conv1/Variable:0")
print(sess.run(x))
sess.close()
