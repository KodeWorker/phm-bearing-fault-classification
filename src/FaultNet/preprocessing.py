import os
import glob
import numpy as np
from scipy.io import loadmat

def ignore_noise(array, n_segments, n_ignore):
    denoised = []
    for idx in range(0, len(array), n_segments):
        segment = array[idx:idx+n_segments]
        denoised.append(segment[n_ignore:-n_ignore].flatten())
    return np.array(denoised)

def get_label(filename):

    if "Normal" in filename:
        label = 0
    elif "B007" in filename:
        label = 1
    elif "B014" in filename:
        label = 2
    elif "B021" in filename:
        label = 3
    elif "IR007" in filename:
        label = 4
    elif "IR014" in filename:
        label = 5
    elif "IR021" in filename:
        label = 6
    elif "OR007" in filename:
        label = 7
    elif "OR014" in filename:
        label = 8
    elif "OR021" in filename:
        label = 9
    else:
        raise ValueError("Filename:{} not valid!".format(filename))
    
    return label

if __name__ == "__main__":
    
    n_samples = 280
    n_segment_datapoints = 1670
    n_ignore_datapoints = 35
    
    data_dir = "../../data/CWRU"
    output_dir = "../../data/CWRU"
    
    filenames = glob.glob(os.path.join(data_dir, "*.mat"))
    #print(filenames)
    CWRU_dataset = np.array([])
    CWRU_labels = np.array([])
    
    for filename in filenames:
        
        mat = loadmat(filename)
        matvalue = None
        for key, value in mat.items():
            if "DE" in key:
                matvalue = value
            else:
                continue
        
        matvalue = matvalue[:n_segment_datapoints*n_samples]
        denoised = ignore_noise(matvalue, n_segment_datapoints, n_ignore_datapoints)
        
        label = get_label(filename)
        labels = np.array([label]*n_samples)
        #print(denoised.shape, labels.shape)  
        #break
        
        if len(CWRU_dataset) == 0:
            CWRU_dataset = denoised
            CWRU_labels = labels
        else:
            CWRU_dataset = np.vstack((CWRU_dataset, denoised))
            CWRU_labels = np.append(CWRU_labels, labels)
    
    #print(CWRU_dataset.shape, CWRU_labels.shape)
    np.save(os.path.join(output_dir, "CWRU_dataset.npy"), CWRU_dataset)
    np.save(os.path.join(output_dir, "CWRU_labels.npy"), CWRU_labels)