import h5py
import os
import ROOT
import numpy as np
import sys
from parse import parse
from utils import *

def read_sample_info(file_path):
    """Reads the sample and cross-section file, returning a list of samples with paths and cross-sections."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.split()
                if len(parts) != 3:
                    print(f"Error: Invalid format in line: {line}")
                    continue
                sample_name, sample_path, xsec = parts
                samples.append((sample_name, sample_path, float(xsec)))
    return samples

def adding_normweight(f_input, xsec=1.0):
    """Add normalized weights to an input file."""
    print(f"Processing file: {f_input}")
    with h5py.File(f_input, "a") as f:
        print(list(f.keys()))
        if 'preselection_eff' in f:
            preselection_eff = f['preselection_eff'][0]
            print(f"preselection_eff: {preselection_eff}")
        else:
            print("Error: 'preselection_eff' not found in file.")
            return

        if xsec > 0.0:
            if 'sys_weights' in list(f.keys()):
                gen_weights = f['sys_weights'][:, 0]
            else:
                gen_weights = f['event_info'][:, 3]

            #gen_weights = np.ones_like(f['sys_weights'][:, 0]) 
            print(f"gen_weights: {gen_weights}")
            rw_factor = xsec * 1000.0 * preselection_eff / np.sum(gen_weights)
            norm_weights = (gen_weights * rw_factor).reshape(-1)
            print(f"weight: {norm_weights}")

            print(f"Total weight: {np.sum(norm_weights)}")

            if 'norm_weights' in f.keys():
                del f['norm_weights']
            f.create_dataset("norm_weights", chunks=True, data=norm_weights, maxshape=None)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <sample_info_file> <max_samples>")
        sys.exit(1)

    sample_info_file = sys.argv[1]
    max_samples = int(sys.argv[2])
    # Directory where sample files are located
    sample_dir = os.path.dirname(sample_info_file)

    # Read the sample information
    samples = read_sample_info(sample_info_file)

    # Process samples
    if max_samples > 0:
        samples = samples[:max_samples]
    
    for sample_name, sample_path, xsec in samples:
    
        # Construct full path if the path is relative
        full_sample_path = os.path.join(sample_dir, sample_path)

        if os.path.exists(full_sample_path):
            print(f"cross-section= {xsec}")
            adding_normweight(full_sample_path, xsec)
        else:
            print(f"Warning: File not found: {full_sample_path}")
