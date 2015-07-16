README for CS 379C Project Code

Authors: Aran Nayebi, Dylan Cable, Richard Wedeen
9 June 2015

All our code has been written in Python.

Libraries needed:

1. Numpy
2. Scipy
3. Theano
4. Keras, a deep learning library in which the LSTM and GRU models are implemented in (can be downloaded from: https://github.com/fchollet/keras). Documentation: http://keras.io/

Explanation of the code:

Data preprocessing code:

1. “read_connectivity.py” reads in the connectivity matrix, and pickles it as a numpy array, entitled, “connectivity.p”.

2. “read_vsoma.py” reads the vsoma data and pickles it as a num_neurons (2022) by num_timesteps (59999) numpy array, entitled, “vsoma.p”.

3. “vsoma_mean0var1.py” makes the mean of the vsoma data to be 0 and its variance to be 1, and pickles this resultant data as a num_neurons (2022) by num_timesteps (59999) numpy array, entitled, “vsoma_rescaled.p”. This is to make the error of the models meaningful (a standard practice in machine learning).

4. “process_data_1000.py” takes “vsoma_rescaled.p” and creates training examples for the GRU and LSTM, such that each training example is a num_neurons (2022) by num_timesteps (in this case, 1000) numpy array. Then, the resultant training examples are pickled as a tensor “vsoma_train_x.p” of size num_train_examples by num_timesteps (1000, in this case) by num_neurons (2022), and their labels are stored as the tensor “vsoma_train_y.p” of size num_train_examples by num_timesteps by num_neurons. Note that the label for each training example (which is a num_neurons by num_timesteps array) is simply the num_neurons by num_timesteps array where each vector of size num_neurons by 1 at a time step t is the label of the vector of num_neurons by 1 at time step t-1 in the training example array. It is important to divide up “vsoma_rescaled.p” in this way, otherwise, one will get memory allocation errors when training either model on “vsoma_rescaled.p” as a single training example.

5. “process_data_3000.py” takes “vsoma_rescaled.p” and creates training examples for the GRU and LSTM, such that each training example is a num_neurons (2022) by num_timesteps (in this case, 3000) numpy array. Everything is the same as in “process_data_1000.py”, just with the num_timesteps being 3000 instead of 1000. However, it turns out that processing the data into 1000 time step chunks ends up with better results than processing it into 3000 time step chunks. Moreover, for the LSTM model, 3000 time steps per training example results in a memory allocation error (but this is not the case for the GRU), so do not run “process_data_3000.py” unless you have more memory than I had access to or you are simply going to train the GRU instead of the LSTM (as I’ve only included “process_data_3000.py” for completeness since I used it for experimentation purposes). 

Code for training the LSTM and GRU models:

1. “theano_script_gru.sh” runs “gru_train_vsoma.py”, the code that trains the GRU, but enables it to use a gpu to train it on. After first invoking the command in terminal, “module load cuda cudasamples” in order to load cuda, I then ran this script to train the GRU on the Stanford Rye Clusters.

2. “theano_script_lstm.sh” runs “lstm_train_vsoma.py”, the code that trains the LSTM, but enables it to use a gpu to train it on. After first invoking the command in terminal, “module load cuda cudasamples” in order to load cuda, I then ran this script to train the LSTM on the Stanford Rye Clusters.

3. “gru_train_vsoma.py” trains the GRU for a specified number of epochs on a specified percentage of the data (“vsoma_train_x.p” and “vsoma_train_y.p”) and validates it on the remaining. It prints the training loss and validation loss after each epoch (the loss is mean squared error (L2 loss) and we use RMS prop). Finally, it saves its weights once it finishes training, to be used as input into “gru_generate_sequence.py”.

4. “lstm_train_vsoma.py” trains the LSTM for a specified number of epochs on a specified percentage of the data (“vsoma_train_x.p” and “vsoma_train_y.p”) and validates it on the remaining. It prints the training loss and validation loss after each epoch (the loss is mean squared error (L2 loss) and we use RMS prop). Finally, it saves its weights once it finishes training, to be used as input into “lstm_generate_sequence.py”.

5. “gru_generate_sequence.py” generates the predicted voltages of the GRU for a specified number of time steps. In my case, I set it to be 3000 since that was enough time steps to plot examples of predicted voltages. However, if you want to set it to something else (that is, as any multiple of 1000 if the data has been processed using “process_data_1000.py”), the code explains how to do that. The output is a pickled num_neurons (2022) by num_timesteps array, named “gru_output.p”, (in my case, since I predicted the voltages of all the neurons at the first 3000 time steps, “gru_output.p” would be of size 2022 by 3000). Hence, to plot the predicted voltages of say, neuron 0, after 1015 time steps, I simply run:

import numpy as np
import pickle
import matplotlib.pyplot as plt
output = pickle.load(open(“output.p”, “wb”));
plt.plot(output[0][:1015])

6. “lstm_generate_sequence.py” generates the predicted voltages of the LSTM for a specified number of time steps. In my case, I set it to be 3000 since that was enough time steps to plot examples of predicted voltages. However, if you want to set it to something else (that is, as any multiple of 1000 if the data has been processed using “process_data_1000.py”), the code explains how to do that. The output is a pickled num_neurons (2022) by num_timesteps array, named “lstm_output.p”, (in my case, since I predicted the voltages of all the neurons at the first 3000 time steps, “lstm_output.p” would be of size 2022 by 3000). Hence, to plot the predicted voltages of say, neuron 0, after 1015 time steps, I can simply run:

import numpy as np
import pickle
import matplotlib.pyplot as plt
output = pickle.load(open(“output.p”, “wb”));
plt.plot(output[0][:1015])