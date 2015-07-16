import os
import time
import struct
import fnmatch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle

vsoma = [];
for dirpath, dirs, files in os.walk('/Users/Aran/Downloads/Costas_Ca_project_Sim2/V_soma'): #enter directory
    for filename in fnmatch.filter(files, '*.bin'):
        with open(os.path.join(dirpath, filename), 'rb') as f:   
            chunk = f.read(4)
            while chunk != '':
                vsoma.append(struct.unpack('f', chunk))
                chunk = f.read(4)
vsoma = np.array(vsoma)
vsoma.shape = [2022, 59999]
pickle.dump(vsoma, open("vsoma.p", "wb"))
#vsoma = pickle.load(open("vsoma.p", "rb")) #to read in the saved data