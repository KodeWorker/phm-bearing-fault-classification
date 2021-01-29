import os
import glob
import pandas as pd

if __name__ == "__main__":
    
    data_dir = "../../data/MAFAULDA"
    output_dir = "../../data/MAFAULDA_X"
    _classes = ["normal",
                "imbalance",
                "horizontal-misalignment",
                "vertical-misalignment",
                "underhang",
                "overhang"]
    
    n_segments = 4096
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for _class in _classes:
        
        filenames = glob.glob(os.path.join(data_dir, _class, "*.csv")) + \
                    glob.glob(os.path.join(data_dir, _class, "*", "*.csv")) + \
                    glob.glob(os.path.join(data_dir, _class, "*", "*", "*.csv"))
        
        print("{}: {} scenarios".format(_class, len(filenames))) # check number of scenarios
        
        class_output_dir = os.path.join(output_dir, _class)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)
        class_sample_count = 0
        
        for filename in filenames:
            
            df = pd.read_csv(filename)
            for idx in range(0, df.shape[0], n_segments):
                _df = df.iloc[idx:idx+n_segments, :]
                if _df.shape[0] == n_segments:
                    class_sample_count += 1
                    _df.to_csv(os.path.join(class_output_dir, "{:05d}.csv".format(class_sample_count)))
                
            #break
        #break
        