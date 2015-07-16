import numpy as np
import pickle

vsoma = pickle.load(open("/Users/Aran/Downloads/vsoma_rescaled.p","rb"))
num_timesteps = 1000
X_train = []
y_train = []
duration = vsoma.shape[1]
print('Duration = '+str(duration))
print('Duration/NumTimeSteps = ' + str(duration/num_timesteps))
cur_it = 0
while cur_it < duration-num_timesteps:
	x = vsoma[:2022,cur_it:cur_it+num_timesteps].T
	y = vsoma[:2022,cur_it+1:cur_it+num_timesteps+1].T
	cur_it += num_timesteps
	X_train.append(x)
	y_train.append(y)
X_train_mat = np.zeros((len(X_train), X_train[0].shape[0], X_train[0].shape[1]))
y_train_mat = np.zeros((len(X_train), X_train[0].shape[0], X_train[0].shape[1]))
for i in xrange(len(X_train)):
	X_train_mat[i] = X_train[i]
	y_train_mat[i] = y_train[i]
	print ('Processed '+str(i)+'/'+str(len(X_train)))
pickle.dump(X_train_mat, open("vsoma_train_x.p", "wb"))
pickle.dump(y_train_mat, open("vsoma_train_y.p", "wb"))