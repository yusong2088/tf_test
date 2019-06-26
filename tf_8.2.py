'''
import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt

with tf.Session() as sess:
    fig,ax=plt.subplots()
    ax.plot(tf.random_normal([100]).eval(),tf.random_normal([100]).eval(),'o')
    ax.set_title('sample random plot for TensorFlow')
    plt.savefig("result.png")
'''
'''
import tensorflow as tf
import numpy as np
#import scipy as sc
#import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib import  pyplot




centers=[(-2,-2),(-2,1.5),(1.5,-2),(2,1.5)]
fig, ax =pyplot.subplots()
data,features=make_blobs(n_samples=200,centers=centers,n_features=2,
                         cluster_std=0.8,shuffle=False,
                         random_state=42)
ax.scatter(np.asarray(data).transpose()[0],np.asarray(data).transpose()[1],marker='o',s=250)
pyplot.show()
'''
from sklearn import datasets
iris=datasets.load_iris()
digits=datasets.load_digits()
print (iris)
print (digits)
print(digits.data),digits.data