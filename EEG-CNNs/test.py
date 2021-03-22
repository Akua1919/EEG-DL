import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

h = 1
v = -2

#prepare data
x_train = np.linspace(-2, 4, 201)                        
noise = np.random.randn(*x_train.shape) * 0.4            
y_train = (x_train - h) ** 2 + v + noise                 

n = x_train.shape[0]

x_train = np.reshape(x_train, (n, 1))                    
y_train = np.reshape(y_train, (n, 1))

#create variable
X = tf.placeholder(tf.float32, [1])                      
Y = tf.placeholder(tf.float32, [1])

with tf.name_scope('W_conv1'):
    h_est = tf.Variable(tf.random_uniform([1], -1, 1), name='h_est')
    print(h_est)

v_est = tf.Variable(tf.random_uniform([1], -1, 1), name='v_est')

saver = tf.train.Saver()

value = (X - h_est) ** 2 + v_est

loss = tf.reduce_mean(tf.square(value - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(50):                             
    for (x, y) in zip(x_train, y_train):
        sess.run(optimizer, feed_dict={X: x, Y: y})
    if epoch % 10 == 0:
        saver.save(sess, './model/model_iter', global_step=epoch)


saver.save(sess, './model/final_model')
h_ = sess.run(h_est)
v_ = sess.run(v_est)

print(h_, v_)