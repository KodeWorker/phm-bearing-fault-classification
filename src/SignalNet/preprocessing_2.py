import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    
    data_dir = "../../data/MAFAULDA_X"
    output_dir = "../../data/MAFAULDA_XX"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filenames = glob.glob(os.path.join(data_dir, "*", "*.csv"))
    
    for filename in tqdm(filenames):
        
        df = pd.read_csv(filename)
        array = df.values[:, 1:]
        reshaped = array.reshape(64,64,8)
        
        _classname = os.path.basename(os.path.dirname(filename))
        _classdir = os.path.join(output_dir, _classname)
        if not os.path.exists(_classdir):
            os.makedirs(_classdir)
    
        _filename = os.path.join(_classdir, os.path.basename(filename).replace(".csv", ".npy"))
        np.save(_filename, reshaped)
        #break