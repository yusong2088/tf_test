import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt

with tf.Session() as sess:
    fig,ax=plt.subplots()
    ax.plot(tf.random_normal([100]).eval(),tf.random_normal([100]).eval(),'o')
    ax.set_title('sample random plot for TensorFlow')
    plt.savefig("result.png")