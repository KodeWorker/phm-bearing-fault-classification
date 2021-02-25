import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def build_argparser():

    parser = ArgumentParser()
    
    parser.add_argument("--data_path", help="path to Mafauldda dataset folder", required=True, type=str)
    parser.add_argument("--output_dir", help="path to output directory", default=None, type=str)
    
    return parser

if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    
    #data_dir = "../../data/MAFAULDA_X"
    #output_dir = "../../data/MAFAULDA_XX"
    data_dir = args.data_dir
    output_dir = args.output_dir
    
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