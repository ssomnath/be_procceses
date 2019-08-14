import sys
import h5py
import pyUSID as usid

from be_sho_fitter import BESHOfitter


def main(input_data_path):

    # Dynamically change mode etc.
    MPI = usid.comp_utils.get_MPI()

    h5_kwargs = {'mode': 'r+'}
    if MPI is not None:
        print('*** Master script called using mpirun ***')
        h5_kwargs.update({'driver': 'mpio', 'comm': MPI.COMM_WORLD})

    h5_f = h5py.File(input_data_path, **h5_kwargs)
    
    h5_main = h5_f['Measurement_000/Channel_000/Raw_Data']
    print(h5_main)

    usid.hdf_utils.print_tree(h5_f)

    proc = BESHOfitter(h5_main, verbose=True)

    proc.set_up_guess()
    h5_guess = proc.do_guess()
    print(h5_guess)

    proc.set_up_fit()
    h5_fit = proc.do_fit()
    print(h5_fit)

    usid.hdf_utils.print_tree(h5_f)

    h5_f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
