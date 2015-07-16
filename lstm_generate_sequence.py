from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pickle
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

hidden_dims = 2022
lstm_hidden_dims = hidden_dims
model = Sequential()
model.add(LSTM(input_dim=hidden_dims, output_dim=lstm_hidden_dims, return_sequences=True))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

#loads the weights that were saved after training
model.load_weights('/Users/Aran/Downloads/cross_validation/1000_timestep_chunks/lstm_my_weights_iter255')
#loads the training data
X_train = pickle.load(open("/Users/Aran/Downloads/cross_validation/1000_timestep_chunks/vsoma_train_x.p","rb"))
print (X_train.shape)
print (X_train[:3].shape) #shape will be (3, 1000, 2022)

#this just picks the first 3000 timesteps (assuming the data 
#has been processed where each training example consists of 1000 timesteps), which is arbitrarily chosen,
#but if you want to change it to just 1000 timesteps simply replace :3 with :1, or if you want to do
#2000 timesteps replace :3 with :2, etc. 
startSeed = X_train[:3]

#reshapes it to be a 1 by 3000 (num_timesteps) by 2022 (num_neurons) tensor (Keras requires its inputs to be a tensor).
#If you wanted the number of timesteps to be 1000, 2000, 4000 etc, simply change the 3000 to your desired number
#of timesteps (has to be a multiple of 1000). 
seedSeq = np.reshape(startSeed, (1, 3000, 2022))
print (seedSeq.shape)

#if we consider at each timestep t as a vector of 2022 (num_neurons) by 1 (representing the voltages of the 2022 neurons at timestep t), 
#then, for each such vector at any timestep t (where in our case t goes from timestep 0 to 3000), then this predicts the voltages
#for timesteps t+1, given the ground truth voltages for timesteps t, t-1, ..., 0. Thus, if t goes from
#0 to 3000, then this predicts the voltages of all 2022 neurons from timesteps 1 to 3001, where at any timestep.
seedSeqNew = model._predict(seedSeq)
print (seedSeqNew.shape)
print (seedSeqNew[0].shape)
output = seedSeqNew[0]
output = output.T
print (output.shape)
pickle.dump(output, open("lstm_output.p", "wb")) #saves the predicted voltages as a 2022 (num_neurons) by num_timesteps array

print ('Finished output eval.')