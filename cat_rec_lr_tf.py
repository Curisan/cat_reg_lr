import numpy as np
import h5py
from lr_utils import load_dataset
import tensorflow as tf

# Loading the data
train_set_x_orig, train_set_y,test_set_x_orig, test_set_y, classes =load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

n = train_set_x.shape[0] # The number of input features
m = train_set_x.shape[1] # The number of training examples

# Initialize the weight and bias with zero
W = tf.Variable(np.zeros((n, 1)), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

# Define placeholder
X = tf.placeholder(dtype=tf.float32, shape=[n,None])
Y = tf.placeholder(dtype=tf.float32, shape=[1,None])

# Forward network
Z = tf.matmul(tf.transpose(W), X)+b
A = tf.sigmoid(Z)

# Compute cost
J = -1/m*(tf.matmul(tf.log(A), tf.transpose(Y))+
          tf.matmul(tf.log(1-A), tf.transpose(1-Y)))

# Optimizer
opt = tf.train.GradientDescentOptimizer(0.005).minimize(J)

sess =  tf.InteractiveSession()
tf.global_variables_initializer().run()

iter = 2000
for i in range(iter):
    opt.run({X:train_set_x, Y:train_set_y})
    if i%100==0:
        print("step"+str(i)+": "+"The cost is "+str(J.eval({X:train_set_x, Y:train_set_y})[0][0]))

# Train accuracy
pred_train = A.eval({X:train_set_x, Y:train_set_y})
corr_train = np.sum(np.equal(pred_train>0.5,train_set_y>0.5))
print("The train accuracy is: "+str(corr_train/m))

# Test accuracy
pred_test = A.eval({X:test_set_x, Y:test_set_y})
corr_test = np.sum(np.equal(pred_test>0.5,test_set_y>0.5))
print("The train accuracy is: "+str(corr_test/test_set_x.shape[1]))
