from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pickle

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

#X_train is a tensor of size (num_train_examples, num_timesteps, num_neurons)
#y_train is a tensor of size (num_train_examples, num_timesteps, num_neurons)

X_train = pickle.load(open("vsoma_train_x.p","rb"))
y_train = pickle.load(open("vsoma_train_y.p","rb"))
print ('Finished loading training data')

print (X_train.shape)
print (y_train.shape)
hidden_dims = 2022
gru_hidden_dims = hidden_dims
model = Sequential()
model.add(GRU(input_dim=hidden_dims, output_dim=gru_hidden_dims, return_sequences=True))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
#model.load_weights('gru_my_weights_iter_n') #uncomment if you want to load weights from a previous iteration

#trains the GRU for 1000 epochs, change nb_epoch to something else if you want to train for less epochs, and
#trains the model on 70% of the data and validates it on the remaining 30%. If you do not want any validation split,
#then just remove it altogether as an argument in model.fit
model.fit(X_train, y_train, batch_size=15, nb_epoch=1000, verbose=1, validation_split=0.3)
model.save_weights("gru_my_weights_iter1000") #saves the weights after training to be used for generation