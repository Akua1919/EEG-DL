#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Hide the Configuration and Warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

# Import the Used Packages: Numpy, Pandas, and Tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Clear the Stack
tf.reset_default_graph()

# The Location of train_data，train_labels，test_data，test_labels
# DataSet Address
DIR = 'Saved_Matlab_Data/'

# Model Saver Address
SAVE = 'Second_Try_Model/'

# Activate a Session
sess = tf.InteractiveSession()

# Read Training Data
train_data = pd.read_csv(DIR + 'training_set_15.csv', header=None)
train_data = np.array(train_data).astype('float32')

# Read Training Labels
train_labels = pd.read_csv(DIR + 'training_label_15.csv', header=None)
train_labels = np.array(train_labels)

# Read Testing Data
test_data = pd.read_csv(DIR + 'valid_set_15.csv', header=None)
test_data = np.array(test_data).astype('float32')

# Read Testing Labels
test_labels = pd.read_csv(DIR + 'valid_label_15.csv', header=None)
test_labels = np.array(test_labels)

# Set Batch Size 64
batch_size = 64
n_batch = train_data.shape[0] // batch_size

# Learning Rate
# lr0 = 0.0001
# lr_decay = 0.99
# lr_step = 500

# Define Placeholders
with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, shape=[None, 60000], name='Input_Data')

    y = tf.placeholder(tf.float32, shape=[None, 3], name='Labels')

    keep_prob = tf.placeholder(tf.float32, name='Keep_Prob')

    x_Reshape = tf.reshape(tensor=x, shape=[-1, 60, 1000, 1], name='Reshape_Data')

# First Convolutional Layer
with tf.name_scope('Convolutional_1'):
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.01), name='W_conv1')

    b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]), name='b_conv1')

    h_conv1 = tf.add(tf.nn.conv2d(x_Reshape, W_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1, name='h_conv1')

    h_conv1_Acti = tf.nn.leaky_relu(h_conv1,name='h_conv1_Acti')

# First Max Pooling Layer
with tf.name_scope('Pooling_1'):
    h_pool1 = tf.nn.max_pool(h_conv1_Acti, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
    
# Second Convolutional Layer
with tf.name_scope('Convolutional_2'):
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.01), name='W_conv2')

    b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), name='b_conv2')

    h_conv2 = tf.add(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME'), b_conv2, name='h_conv2')

    h_conv2_Acti = tf.nn.leaky_relu(h_conv2, name='h_conv2_Acti')
    
# Second Max Pooling Layer
with tf.name_scope('Pooling_2'):
    h_pool2 = tf.nn.max_pool(h_conv2_Acti, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
    
# Flatten Layer
with tf.name_scope('Flatten'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 15*250*64], name='h_pool2_flat')


# First Fully Connected Layer
with tf.name_scope('Fully_Connected_1'):
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[15*250*64, 512], stddev=0.01), name='W_fc1')

    b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]), name='b_fc1')

    h_fc1 = tf.matmul(h_pool2_flat, W_fc1, name='h_fc1') + b_fc1

    h_fc1_Acti = tf.nn.leaky_relu(h_fc1, name='h_fc1_Acti')

    h_fc1_drop = tf.nn.dropout(h_fc1_Acti, keep_prob, name='h_fc1_drop')

# Second Fully Connected Layer
with tf.name_scope('Output_Layer'):
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[512, 3], stddev=0.01), name='W_fc2')

    b_fc2 = tf.Variable(tf.constant(0.01, shape=[3]), name='b_fc2')

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='prediction')

# Calculate loss
loss = tf.reduce_mean(tf.square(y - prediction), name='loss')
tf.summary.scalar('loss', loss)

# Optimize
train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
#tf.summary.scalar('keep_prob', keep_prob)

with tf.name_scope('Evalution'):
    # Calculate Each Task Accuracy
    with tf.name_scope('Each_Class_accuracy'):
        # Task 1 Accuracy
        with tf.name_scope('T1_accuracy'):
            # Number of Classified Correctly
            y_T1 = tf.equal(tf.argmax(y, 1), 0)
            prediction_T1 = tf.equal(tf.argmax(prediction, 1), 0)
            T1_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T1), tf.float32))

            # Number of All the Test Samples
            T1_all_Num = tf.reduce_sum(tf.cast(y_T1, tf.float32))

            # Task 1 Accuracy
            T1_accuracy = tf.divide(T1_Corrected_Num, T1_all_Num, name='T1_accuracy')
            tf.summary.scalar('T1_accuracy', T1_accuracy)

            T1_TP = T1_Corrected_Num
            T1_TN = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T1), tf.math.logical_not(prediction_T1)), tf.float32))
            T1_FP = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T1), prediction_T1), tf.float32))
            T1_FN = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, tf.math.logical_not(prediction_T1)), tf.float32))

            # Task 1 Precision
            T1_Precision = tf.divide(T1_TP, T1_TP + T1_FP, name='T1_Precision')
            tf.summary.scalar('T1_Precision', T1_Precision)
            # Task 1 Recall
            T1_Recall = tf.divide(T1_TP, T1_TP + T1_FN, name='T1_Recall')
            tf.summary.scalar('T1_Recall', T1_Recall)
            # Task 1 F_Score
            T1_F_Score = tf.divide(2*T1_Precision*T1_Recall, T1_Precision+T1_Recall, name='T1_F_Score')
            tf.summary.scalar('T1_F_Score', T1_F_Score)

        # Task 2 Accuracy
        with tf.name_scope('T2_accuracy'):
            # Number of Classified Correctly
            y_T2 = tf.equal(tf.argmax(y, 1), 1)
            prediction_T2 = tf.equal(tf.argmax(prediction, 1), 1)
            T2_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T2), tf.float32))

            # Number of All the Test Samples
            T2_all_Num = tf.reduce_sum(tf.cast(y_T2, tf.float32))

            # Task 2 Accuracy
            T2_accuracy = tf.divide(T2_Corrected_Num, T2_all_Num, name='T2_accuracy')
            tf.summary.scalar('T2_accuracy', T2_accuracy)

            T2_TP = T2_Corrected_Num
            T2_TN = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T2), tf.math.logical_not(prediction_T2)), tf.float32))
            T2_FP = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T2), prediction_T2), tf.float32))
            T2_FN = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, tf.math.logical_not(prediction_T2)), tf.float32))

            # Task 2 Precision
            T2_Precision = tf.divide(T2_TP, T2_TP + T2_FP, name='T2_Precision')
            tf.summary.scalar('T2_Precision', T2_Precision)
            # Task 2 Recall
            T2_Recall = tf.divide(T2_TP, T2_TP + T2_FN, name='T2_Recall')
            tf.summary.scalar('T2_Recall', T2_Recall)
            # Task 2 F_Score
            T2_F_Score = tf.divide(2*T2_Precision*T2_Recall, T2_Precision+T2_Recall, name='T2_F_Score')
            tf.summary.scalar('T2_F_Score', T2_F_Score)

        # Task 3 Accuracy
        with tf.name_scope('T3_accuracy'):
            # Number of Classified Correctly
            y_T3 = tf.equal(tf.argmax(y, 1), 2)
            prediction_T3 = tf.equal(tf.argmax(prediction, 1), 2)
            T3_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T3), tf.float32))

            # Number of All the Test Samples
            T3_all_Num = tf.reduce_sum(tf.cast(y_T3, tf.float32))

            # Task 3 Accuracy
            T3_accuracy = tf.divide(T3_Corrected_Num, T3_all_Num, name='T3_accuracy')
            tf.summary.scalar('T3_accuracy', T3_accuracy)

            T3_TP = T3_Corrected_Num
            T3_TN = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T3), tf.math.logical_not(prediction_T3)), tf.float32))
            T3_FP = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_T3), prediction_T3), tf.float32))
            T3_FN = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, tf.math.logical_not(prediction_T3)), tf.float32))

            # Task 3 Precision
            T3_Precision = tf.divide(T3_TP, T3_TP + T3_FP, name='T3_Precision')
            tf.summary.scalar('T3_Precision', T3_Precision)
            # Task 3 Recall
            T3_Recall = tf.divide(T3_TP, T3_TP + T3_FN, name='T3_Recall')
            tf.summary.scalar('T3_Recall', T3_Recall)
            # Task 3 F_Score
            T3_F_Score = tf.divide(2*T3_Precision*T3_Recall, T3_Precision+T3_Recall, name='T3_F_Score')
            tf.summary.scalar('T3_F_Score', T3_F_Score)

    # Calculate the Confusion Matrix
    with tf.name_scope("Confusion_Matrix"):
        with tf.name_scope("T1_Label"):
            T1_T1 = T1_Corrected_Num
            T1_T2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T2), tf.float32))
            T1_T3 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T3), tf.float32))

            T1_T1_percent = tf.divide(T1_T1, T1_all_Num, name='T1_T1_percent')
            T1_T2_percent = tf.divide(T1_T2, T1_all_Num, name='T1_T2_percent')
            T1_T3_percent = tf.divide(T1_T3, T1_all_Num, name='T1_T3_percent')

            tf.summary.scalar('T1_T1_percent', T1_T1_percent)
            tf.summary.scalar('T1_T2_percent', T1_T2_percent)
            tf.summary.scalar('T1_T3_percent', T1_T3_percent)

        with tf.name_scope("T2_Label"):
            T2_T1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T1), tf.float32))
            T2_T2 = T2_Corrected_Num
            T2_T3 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T3), tf.float32))

            T2_T1_percent = tf.divide(T2_T1, T2_all_Num, name='T2_T1_percent')
            T2_T2_percent = tf.divide(T2_T2, T2_all_Num, name='T2_T2_percent')
            T2_T3_percent = tf.divide(T2_T3, T2_all_Num, name='T2_T3_percent')

            tf.summary.scalar('T2_T1_percent', T2_T1_percent)
            tf.summary.scalar('T2_T2_percent', T2_T2_percent)
            tf.summary.scalar('T2_T3_percent', T2_T3_percent)

        with tf.name_scope("T3_Label"):
            T3_T1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T1), tf.float32))
            T3_T2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T2), tf.float32))
            T3_T3 = T3_Corrected_Num

            T3_T1_percent = tf.divide(T3_T1, T3_all_Num, name='T3_T1_percent')
            T3_T2_percent = tf.divide(T3_T2, T3_all_Num, name='T3_T2_percent')
            T3_T3_percent = tf.divide(T3_T3, T3_all_Num, name='T3_T3_percent')

            tf.summary.scalar('T3_T1_percent', T3_T1_percent)
            tf.summary.scalar('T3_T2_percent', T3_T2_percent)
            tf.summary.scalar('T3_T3_percent', T3_T3_percent)

    with tf.name_scope('Global_Evalution_Metrics'):
        # Global Average Accuracy - Simple Algorithm
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        Global_Average_Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Global_Average_Accuracy')
        tf.summary.scalar('Global_Average_Accuracy', Global_Average_Accuracy)

        with tf.name_scope('Kappa_Metric'):
            Test_Set_Num = T1_all_Num + T2_all_Num + T3_all_Num

            Actual_T1 = T1_all_Num
            Actual_T2 = T2_all_Num
            Actual_T3 = T3_all_Num

            Prediction_T1 = T1_T1 + T2_T1 + T3_T1         
            Prediction_T2 = T1_T2 + T2_T2 + T3_T2         
            Prediction_T3 = T1_T3 + T2_T3 + T3_T3         

            p0 = (T1_T1 + T2_T2 + T3_T3) / Test_Set_Num   
            pe = (Actual_T1*Prediction_T1 + Actual_T2*Prediction_T2 + Actual_T3*Prediction_T3) / \
                 (Test_Set_Num*Test_Set_Num)

            Kappa_Metric = tf.divide(p0 - pe, 1 - pe, name='Kappa_Metric')
            tf.summary.scalar('Kappa_Metric', Kappa_Metric)

        with tf.name_scope('Micro_Averaged_Evalution'):
            TP_all = T1_TP + T2_TP + T3_TP
            TN_all = T1_TN + T2_TN + T3_TN
            FP_all = T1_FP + T2_FP + T3_FP
            FN_all = T1_FN + T2_FN + T3_FN

            Micro_Global_Precision = tf.divide(TP_all, TP_all + FP_all, name='Micro_Global_Precision')
            tf.summary.scalar('Micro_Global_Precision', Micro_Global_Precision)

            Micro_Global_Recall = tf.divide(TP_all, TP_all + FN_all, name='Micro_Global_Recall')
            tf.summary.scalar('Micro_Global_Recall', Micro_Global_Recall)

            Micro_Global_F1_Score = tf.divide(2*Micro_Global_Precision*Micro_Global_Recall, Micro_Global_Precision+Micro_Global_Recall, name='Micro_Global_F1_Score')
            tf.summary.scalar('Micro_Global_F1_Score', Micro_Global_F1_Score)

        with tf.name_scope('Macro_Averaged_Evalution'):
            Macro_Global_Precision = tf.divide(T1_Precision + T2_Precision + T3_Precision, 3.0, name='Macro_Global_Precision')
            tf.summary.scalar('Macro_Global_Precision', Macro_Global_Precision)

            Macro_Global_Recall = tf.divide(T1_Recall + T2_Recall + T3_Recall, 3.0, name='Macro_Global_Recall')
            tf.summary.scalar('Macro_Global_Recall', Macro_Global_Recall)

            Macro_Global_F1_Score = tf.divide(T1_F_Score + T2_F_Score + T3_F_Score, 3.0, name='Macro_Global_F1_Score')
            tf.summary.scalar('Macro_Global_F1_Score', Macro_Global_F1_Score)

# Merge all the summaries
merged = tf.summary.merge_all()

# Initialize all the variables
sess.run(tf.global_variables_initializer())

# Start a saver to save the trained model
saver = tf.train.Saver()

# Summary the Training and Test Processing
train_writer = tf.summary.FileWriter(SAVE + 'train_Writer', sess.graph)
test_writer  = tf.summary.FileWriter(SAVE + 'test_Writer')

for epoch in range(2000):
    for batch_index in range(n_batch):
        random_batch = random.sample(range(train_data.shape[0]), batch_size)
        batch_xs = train_data[random_batch]
        batch_ys = train_labels[random_batch]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.50})

    # Accuracy on Training Set
    train_acc = 0.0
    train_loss = 0.0
    for batch_index in range(20):
        random_batch = random.sample(range(train_data.shape[0]), 150)
        batch_xs = train_data[random_batch]
        batch_ys = train_labels[random_batch]
        dtrain_acc, dtrain_loss = sess.run([Global_Average_Accuracy, loss], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_acc = train_acc + dtrain_acc
        train_loss = train_loss + dtrain_loss
    train_acc = train_acc / 20

    # Accuracy on Test Set
    test_acc = 0.0
    test_loss = 0.0
    for batch_index in range(20):
        random_batch = random.sample(range(train_data.shape[0]), 150)
        batch_xs = train_data[random_batch]
        batch_ys = train_labels[random_batch]
        test_summary, dtest_acc, dtest_loss = sess.run([merged, Global_Average_Accuracy, loss], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_acc = test_acc + dtest_acc
        test_loss = test_loss + dtest_loss
    test_acc = test_acc / 20

    test_writer.add_summary(test_summary, epoch)
    
    # Show the Model Capability
    with open("Log/log_2nd.out","a") as f:
        f.write("Iter " + str(epoch) + ", Training Accuracy: " + str(train_acc) + ", Training Loss: "+ str(train_loss)+", Testing Accuracy: " + str(test_acc) + ", Testing Loss: "+ str(test_loss) + '\n')

    # Save the Model Every 100 Epoches
    if epoch % 600 == 0:
        saver.save(sess, save_path=SAVE + 'Model_Saver/', global_step=epoch)

    # if epoch == 1999:
    #     output_prediction = sess.run(prediction, feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
    #     np.savetxt(SAVE + "prediction.csv", output_prediction, delimiter=",")
    #     np.savetxt(SAVE + "labels.csv", test_labels, delimiter=",")

train_writer.close()
test_writer.close()
sess.close()
