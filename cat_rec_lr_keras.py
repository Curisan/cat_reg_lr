import numpy as np
import h5py
from lr_utils import load_dataset
from keras.models import Sequential
from keras.layers import Dense, Activation

# Loading the data
train_set_x_orig, train_set_y,test_set_x_orig, test_set_y, classes =load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
train_set_x = train_set_x.T
test_set_x = test_set_x.T
train_set_y = train_set_y.T
test_set_y = test_set_y.T

m = train_set_x.shape[0] # The number of training examples
n = train_set_x.shape[1] # The number of input features

model = Sequential()
model.add(Dense(output_dim=1, input_shape=(n,)))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])

model.fit(train_set_x, train_set_y, nb_epoch=2000, batch_size=m)
loss_and_metrics = model.evaluate(train_set_x, train_set_y, batch_size=m)
print("The train accuracy is: "+str(loss_and_metrics[1]))
loss_and_metrics = model.evaluate(test_set_x, test_set_y, batch_size=m)
print("The test accuracy is: "+str(loss_and_metrics[1]))
