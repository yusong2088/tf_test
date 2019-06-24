'''import tensorflow as tf

input1=tf.constant(1)
print(input1)
input2=tf.Variable(2,tf.int32)
print(input2)
input2=input1

sess=tf.Session()
print(sess.run(input2))
'''

'''
import tensorflow as tf
input1=tf.placeholder(tf.int32)
input2=tf.placeholder(tf.int32)

output=tf.add(input1,input2)

sess=tf.Session()
print(sess.run(output,feed_dict={input1:[1],input2:[2]}))
'''

import tensorflow as tf
import numpy as np

inputX=np.random.rand(3000,1)
noise=np.random.normal(0,0.05,inputX.shape)
outputY=inputX*4+1+noise

weight1=tf.Variable(np.random.rand(inputX.shape[1],4))
bias1=tf.Variable(np.random.rand(inputX.shape[1],4))

x1=tf.placeholder(tf.float64,[None,1])
y1_=tf.matmul(x1,weight1)+bias1

y=tf.placeholder(tf.float64,[None,1])

loss=tf.reduce_mean(tf.reduce_sum(tf.square(y1_-y)),reduction_indices=[1])

train=tf.train.GradientDescentOptimizer(0.25).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train,feed_dict={x1:inputX,y:outputY})

print(weight1.eval(sess))
print("-----------------")
print(bias1.eval(sess))
print("-----------结果是----------")
x_data=np.array([[1.],[2.],[3.]])
print(sess.run(y1_,feed_dict={x1:x_data}))