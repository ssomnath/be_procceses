import sys
import h5py
import pyUSID as usid

from process import Process
from fitter import Fitter
from be_sho_fitter import BESHOfitter


def main(input_data_path):

    # Dynamically change mode etc.
    MPI = usid.comp_utils.get_MPI()

    h5_kwargs = {'mode': 'r+'}
    mpi_rank = 0

    if MPI is not None:
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        if mpi_rank == 0:
            print('*** Master script called using mpirun ***')
        h5_kwargs.update({'driver': 'mpio', 'comm': MPI.COMM_WORLD})

    if mpi_rank == 0:
        print('h5py kwargs: {}'.format(h5_kwargs))

    h5_f = h5py.File(input_data_path, **h5_kwargs)
    
    h5_main = h5_f['Measurement_000/Channel_000/Raw_Data']

    if mpi_rank == 0:
        print(h5_main)
        usid.hdf_utils.print_tree(h5_f)

    proc = BESHOfitter(h5_main, verbose=True)

    """
    if mpi_rank == 0:
        print('*** Instantiated the fitter ***')

    proc.set_up_guess()

    if mpi_rank == 0:
        print('*** Set up the guess ***')

    if MPI is not None:
        MPI.COMM_WORLD.barrier()

    h5_guess = proc.do_guess()

    if mpi_rank == 0:
        print('*** Guess complete ***')

    if mpi_rank == 0:
        print(h5_guess)
        usid.hdf_utils.print_tree(h5_f)

    proc.set_up_fit()

    if MPI is not None:
        MPI.COMM_WORLD.barrier()

    if mpi_rank == 0:
        print('*** Set up fit ***')

    h5_fit = proc.do_fit()

    if mpi_rank == 0:
        print('*** Fit complete ***')

    if mpi_rank == 0:
        print(h5_fit)
        usid.hdf_utils.print_tree(h5_f)
    """
    if MPI is not None:
        MPI.COMM_WORLD.barrier()

    print('Rank {}: about to close the file'.format(mpi_rank))

    h5_f.close()

    print('Rank {}: exiting'.format(mpi_rank))


if __name__ == "__main__":
    main(sys.argv[1])
