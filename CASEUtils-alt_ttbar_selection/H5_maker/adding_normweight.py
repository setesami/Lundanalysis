import h5py
import os
import ROOT
import numpy as np
import sys
from parse import parse
from utils import *


def adding_normweight(f_input, xsec = 1.0, max_files = -1):

    print(f_input)



    with h5py.File(f_input, "a") as f:
        print(list(f.keys()))
        if 'preselection_eff' in f:
          preselection_eff = f['preselection_eff'][0]
          print(f"preselection_eff: {preselection_eff}")


        if (xsec > 0.):
            if('sys_weights' in list(f.keys())):
                gen_weights = f['sys_weights'][:,0]
            else:
                gen_weights = f['event_info'][:,3]

            rw_factor = xsec * 1000. * preselection_eff / np.sum(gen_weights)

            norm_weights  = (gen_weights * rw_factor).reshape(-1)
            print("Total weight is %s" % np.sum(norm_weights))

            if('norm_weights' in f.keys()):
                del f['norm_weights']
            f.create_dataset("norm_weights", chunks = True, data= norm_weights, maxshape = None)

if(__name__ == "__main__"):
    print(sys.argv)
    adding_normweight(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))

