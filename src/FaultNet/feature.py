import numpy as np

def mean(data,no_elements):
    X=np.zeros((data.shape[0],data.shape[1]))
    for i in range(data.shape[1]-no_elements+1):
        X[:,i]=np.mean(data[:,i:i+no_elements],axis=1)
    return X.astype(np.float16)
    
def median(data,no_elements):
    X=np.zeros((data.shape[0],data.shape[1]))
    for i in range(data.shape[1]-no_elements+1):
        X[:,i]=np.median(data[:,i:i+no_elements],axis=1)
    return X.astype(np.float16)
    
def sig_image(data,size):
    X=np.zeros((data.shape[0],size,size))
    for i in range(data.shape[0]):
        X[i]=(data[i,:].reshape(size,size))
    return X.astype(np.float16)