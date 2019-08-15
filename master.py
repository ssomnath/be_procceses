import sys
import time
import h5py
import pyUSID as usid

from be_sho_fitter import BESHOfitter


def main(input_data_path):

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

    proc = BESHOfitter(h5_main, verbose=False)

    if mpi_rank == 0:
        print('*** Instantiated the fitter ***')

    proc.set_up_guess()

    if mpi_rank == 0:
        print('*** Finished set up of guess ***')

    if MPI is not None:
        MPI.COMM_WORLD.barrier()

    if mpi_rank == 0:
        t0 = time.time()

    _ = proc.do_guess()

    if mpi_rank == 0:
        print('*** Guess completed in ' + usid.io_utils.format_time(time.time() - t0) + ' ***')

    proc.set_up_fit()

    if MPI is not None:
        MPI.COMM_WORLD.barrier()

    if mpi_rank == 0:
        t0 = time.time()
        print('*** Finished set up of fit ***')

    _ = proc.do_fit()

    if mpi_rank == 0:
        print('*** Fit completed in ' + usid.io_utils.format_time(time.time() - t0) + ' ***')

    if MPI is not None:
        MPI.COMM_WORLD.barrier()

    print('Rank {}: about to close the file'.format(mpi_rank))

    h5_f.close()

    print('Rank {}: exiting'.format(mpi_rank))


if __name__ == "__main__":
    main(sys.argv[1])
