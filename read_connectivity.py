import numpy as np
import pickle

x = np.loadtxt('/Users/Aran/Downloads/Costas_Ca_project_Sim2/Connectivity/loc_con_ASCII.txt')
pickle.dump(x, open("connectivity.p", "wb"))
#connectivity = pickle.load(open("connectivity.p", "rb")) #to read in the saved data