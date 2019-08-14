import sys
# sys.path.append(r'C:\Users\Suhas\PycharmProjects\pyUSID')
# sys.path.append(r'C:\Users\Suhas\PycharmProjects\pycroscopy')
import os
import h5py
import shutil
import numpy as np
import pyUSID as usid

def main(orig_data_path):
    
    from be_sho_fitter import BESHOfitter

    # orig_data_path = r'C:\Users\Suhas\PycharmProjects\pyUSID\data\BEPS_small.h5'

    # Dynamically change mode etc.
    mpi_run = False
    h5_kwargs = {'mode': 'r+'}
    if mpi_run:
        h5_kwargs = {}
    h5_f = h5py.File(input_data_path, **kwargs)
    
    h5_main = h5_f['Measurement_000/Channel_000/Raw_Data']
    print(h5_main)

    usid.hdf_utils.print_tree(h5_f)

    proc = BESHOfitter(h5_main, verbose=True)
    proc.set_up_guess()
    h5_guess = proc.do_guess()

    proc.set_up_fit()
    proc.do_fit()
    usid.hdf_utils.print_tree(h5_f)

    h5_f.close()

if __name__ == "__main__":
    main(sys.argv[1:])