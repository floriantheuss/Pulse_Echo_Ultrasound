import numpy as np
import matplotlib.pyplot as plt
from time import time



# # file = 'C:\\Users\\Doodleous\\Dropbox\\Mn3Ge2104D\\Field\\data_7_7_2021\\run_2_50_26_pm_199MHz\\all_data.txt'
file_load = 'C:\\Users\\Doodleous\\Dropbox\\Mn3Ge2104D\\Field\\data_7_7_2021\\run_6_21_50_pm_199MHz\\all_data.txt'

t1 = time()
y_all = []
y_individual = []
indicator = 0

f = open(file_load, 'r')
for _ in np.arange(3):
    f.readline()
for line in f:
    line = line.strip()
    line = line.split()
    if len( line ) >= 1:
        if line[0] == 't0':
            y_all.append( np.array(y_individual) )
            y_individual = []
        elif line[0] == 'delta':
            pass
        elif line[0] == '':
            pass
        elif line[0] == 'time':
            pass
        else:
            y_individual.append( float(line[-1]) )
    else:
        pass
    indicator+=1
    print (indicator)

print (time()-t1)
print (len(y_all))

# t2 = time()
# data = np.genfromtxt(file_load)
# print (data)
# print (time()-t2)




y_all = np.array(y_all)

print (len(y_all))

# n = 5000
# plt.figure()
# plt.plot(np.arange(len(y_all[n])), y_all[n])
# plt.show()

file_save = 'C:\\Users\\Doodleous\\Dropbox\\Mn3Ge2104D\\Field\\data_7_7_2021\\run_6_21_50_pm_199MHz\\raw_data.txt'

# t1 = time()
# data = np.loadtxt(file_save)
# print(time()-t1)
np.savetxt(file_save, y_all)

# n = 5000
# plt.figure()
# plt.plot(np.arange(len(data[n])), data[n])
# plt.show()

