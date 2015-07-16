import numpy as np
import matplotlib.pyplot as plt
import pickle

vsoma = pickle.load(open("vsoma.p", "rb"))
m = vsoma.shape[0]*vsoma.shape[1]
mean_v = np.sum(vsoma)/(m)
vsoma -= mean_v
var = np.sum(vsoma**2)/(m)
vsoma /= np.sqrt(var)
pickle.dump(vsoma, open("vsoma_rescaled.p", "wb"))
