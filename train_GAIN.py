#%% Packages

import sys
import numpy as np

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

#%% System Parameters
# 1. Mini batch size
mb_size = 32
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

#%% Data

# Data generation
data_file= str(sys.argv[0])
Data = np.loadtxt(data_file, delimiter=",",skiprows=1)

# Parameters
No = len(Data)
Dim = len(Data[0,:])

# Hidden state dimensions
H_Dim1 = Dim
H_Dim2 = Dim

# # Normalization (0 to 1)
# Min_Val = np.zeros(Dim)
# Max_Val = np.zeros(Dim)

# for i in range(Dim):
#     Min_Val[i] = np.min(Data[:,i])
#     Data[:,i] = Data[:,i] - np.min(Data[:,i])
#     Max_Val[i] = np.max(Data[:,i])
#     Data[:,i] = Data[:,i] / (np.max(Data[:,i]) + 1e-6)    

GAIN_MSEs = []
for _ in range(10): 

    #%% Missing introducing
    p_miss_vec = p_miss * np.ones((Dim,1)) 
       
    Missing = np.zeros((No,Dim))

    for i in range(Dim):
        A = np.random.uniform(0., 1., size = [len(Data),])
        B = A > p_miss_vec[i]
        Missing[:,i] = 1.*B
 
    #%% Train Test Division    
       
    idx = np.random.permutation(No)

    Train_No = int(No * train_rate)
    Test_No = No - Train_No
        
    # Train / Test Features
    trainX = Data[idx[:Train_No],:]
    testX = Data[idx[Train_No:],:]

    # Train / Test Missing Indicators
    trainM = Missing[idx[:Train_No],:]
    testM = Missing[idx[Train_No:],:]

    # Export indices and missing indicators for benchmarks
    np.savetxt('spam/train_data_{}.csv'.format(_),trainX, delimiter=',')
    np.savetxt('spam/test_data_{}.csv'.format(_),testX, delimiter=',')

    np.savetxt('spam/train_missing_{}.csv'.format(_),trainM, delimiter=',')
    np.savetxt('spam/test_missing_{}.csv'.format(_),testM, delimiter=',')
    
    # Scale 0 to 1

    trainX = min_max_scaler.fit_transform(trainX)
    testX = min_max_scaler.transform(testX)

    execfile("GAIN.py")

print('GAIN Mean Test RMSE: {:.4}'.format(np.mean(np.sqrt(GAIN_MSEs))))
print('GAIN SD: {:.4}'.format(np.std(np.sqrt(GAIN_MSEs))))