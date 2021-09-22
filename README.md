# Inferring Network Dynamics from Simulated Extracellular Electrophysiological Data

This is code I wrote for a course project for CS 379C: Computational Models of the Neocortex, taught by Tom Dean in Spring Quarter 2015. Please refer to README.txt for explanations of the code.

Project Abstract: We compare the performance of two different types of recurrent neural networks (RNNs) for the task of inferring the extracellular voltage of each neuron at a given timestep, when provided with partial information about the simulated extracellular electrophysiological data of a population of 2022 neurons over 6 seconds. In particular, we focus on RNNs that have a sophisticated gating mechanism, namely, the Long Short-Term Memory (LSTM) network and the recently introduced Gated Recurrent Unit (GRU). When trained on 70 percent of the timesteps (and validated on the remaining 30 percent), our results indicate that the predicted voltages for each neuron of the GRU network were significantly more plausible than the predictions of the LSTM, when compared to the ground truth voltages. This provides encouraging evidence that RNNs (specifically the GRU) can be considered a promising approach to inferring network dynamics in large populations of neurons. We also discuss some preliminary results towards the goal of inferring synaptic weights from this electrophysiological data as well.

[Full Paper](https://anayebi.github.io/files/projects/CS_379C_Final_Project_Writeup.pdf)
