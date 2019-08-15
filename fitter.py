import numpy as np
import sys
from warnings import warn

sys.path.append(r'C:\Users\Suhas\PycharmProjects\pyUSID')

# Until I figure out what in the Process class is killing things....
from usi_data import USIDataset
from process import Process


class Fitter(Process):
    
    def __init__(self, h5_main, variables=None, **kwargs):
        super(Fitter, self).__init__(h5_main, **kwargs)
        if self.verbose:
            print('Rank {} at Fitter: Just finished coming out of Process'.format(self.mpi_rank))
        
        # Validate other arguments / kwargs here:
        if variables is not None:
            if not np.all(np.isin(variables, self.h5_main.spec_dim_labels)):
                raise ValueError('Provided dataset does not appear to have the spectroscopic dimension(s) that need to be fitted')

        self.process_name = None
                
        self.guess = None
        self.fit = None
        
        self.h5_guess = None
        self.h5_fit = None
        self.h5_results_grp = None
        
        self._is_guess = True
        self.__mode = 0  # 0 for Guess pending, 1 for Fit pending, 2 for fit complete

        if self.verbose:
            print('Rank {} at Fitter: Just finished init'.format(self.mpi_rank))
        
    def _read_guess_chunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset.

        Parameters
        -----
        None

        Returns
        --------

        """
        curr_pixels = self._get_pixels_in_current_batch()
        self.guess = self.h5_guess[curr_pixels, :]

        if self.verbose and self.mpi_rank == 0:
            print('Guess of shape: {}'.format(self.guess.shape))
            
    def _write_results_chunk(self):
        """
        Writes the provided guess or fit results into appropriate datasets.
        Given that the guess and fit datasets are relatively small, we should be able to hold them in memory just fine

        Parameters
        ---------
        is_guess : bool, optional
            Default - False
            Flag that differentiates the guess from the fit
        """
        statement = 'guess'

        if self._is_guess:
            targ_dset = self.h5_guess
            source_dset = self.guess
        else:
            statement = 'fit'
            targ_dset = self.h5_fit
            source_dset = self.fit
            
        curr_pixels = self._get_pixels_in_current_batch()

        if self.verbose and self.mpi_rank == 0:
            print('Writing data of shape: {} and dtype: {} to position range: {} '
                  'in HDF5 dataset:{}'.format(source_dset.shape,
                                              source_dset.dtype,
                                              [curr_pixels[0],curr_pixels[-1]],
                                              targ_dset))
        targ_dset[curr_pixels, :] = source_dset
        
    def _create_guess_datasets(self):
        """
        Model specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.

        The guess dataset will NOT be populated here but will be populated by the __setData function
        The fit dataset should NOT be populated here unless the user calls the optimize function.

        Parameters
        --------
        None

        Returns
        -------
        None

        """
        raise NotImplementedError('Please override the _create_guess_datasets specific to your model')

    def _create_fit_datasets(self):
        """
        Model specific call that will write the h5 group, fit dataset, corresponding spectroscopic datasets and also
        link the fit dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.

        The fit dataset will NOT be populated here but will be populated by the __setData function
        The guess dataset should NOT be populated here unless the user calls the optimize function.

        Parameters
        --------
        None

        Returns
        -------
        None

        """
        raise NotImplementedError('Please override the _create_fit_datasets specific to your model')
        
    def _get_existing_datasets(self):
        self.h5_guess = self.h5_results_grp['Guess']
        
        try:
            self._h5_status_dset = self.h5_results_grp[self._status_dset_name]
        except KeyError:
            warn('status dataset not created yet')
            self._h5_status_dset = None
            
        try:
            self.h5_fit = self.h5_results_grp['Fit']
        except KeyError:
            self.h5_fit = None
            if not self._is_guess:
                self._create_fit_datasets()
        
    def _check_for_old_guess(self):
        """
        Checks just the status dataset

        Returns
        -------

        """
        # return partial_dsets, completed_dsets (reverse as what Process returns below!)
        return super(Fitter, self)._check_for_duplicates()
    
    def _check_for_old_fit(self):
        """
        Returns three lists of h5py.Dataset objects where the group contained:
            1. Completed guess only
            2. Partial Fit
            3. Completed Fit

        Returns
        -------

        """
        completed_fits, partial_fits = super(Fitter, self)._check_for_duplicates()
        # return completed_guess, partial_fits, completed_fits
        pass
        
    def do_guess(self, *args, override=False, **kwargs):
        self.h5_results_grp = super(Fitter, self).compute(override=override)
        return USIDataset(self.h5_results_grp['Guess']) 
    
    def do_fit(self, *args, override=False, **kwargs):
        self.h5_results_grp = super(Fitter, self).compute(override=override)
        return USIDataset(self.h5_results_grp['Fit']) 
    
    # DBJV46
    
    def _reformat_results(self, results, strategy='wavelet_peaks'):
        """
        Model specific restructuring / reformatting of the parallel compute results

        Parameters
        ----------
        results : array-like
            Results to be formatted for writing
        strategy : str
            The strategy used in the fit.  Determines how the results will be reformatted.
            Default 'wavelet_peaks'

        Returns
        -------
        results : numpy.ndarray
            Formatted array that is ready to be writen to the HDF5 file 

        """
        return np.array(results)

    def set_up_guess(self, h5_partial_guess=None):

        # Set up the parms dict so everything necessary for checking previous guess / fit is ready
        self._is_guess = True
        self._status_dset_name = 'completed_guess_positions'
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        if self.verbose and self.mpi_rank == 0:
            print('Groups with Guess in:\nCompleted: {}\nPartial:{}'.format(
                self.duplicate_h5_groups, self.partial_h5_groups))

        self._unit_computation = super(Fitter, self)._unit_computation
        self._create_results_datasets = self._create_guess_datasets


    def set_up_fit(self, h5_partial_fit=None, h5_guess=None):

        self._is_guess = False

        self._map_function = None
        self._unit_computation = None
        self._create_results_datasets = self._create_fit_datasets

        # Case 1: Fit already complete or partially complete. This is similar to a partial process. Leave as is
        self._status_dset_name = 'completed_fit_positions'
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()
        if self.verbose and self.mpi_rank == 0:
            print('Checking on partial / completed fit datasets')
            print(
                'Completed results groups:\n{}\nPartial results groups:\n{}'.format(
                    self.duplicate_h5_groups,
                    self.partial_h5_groups))

        # Case 2: Fit neither partial / completed. Search for guess. Most popular scenario:
        if len(self.duplicate_h5_groups) == 0 and len(
                self.partial_h5_groups) == 0:
            if self.verbose and self.mpi_rank == 0:
                print('No fit datasets found. Looking for Guess datasets')
            # Change status dataset name back to guess to check for status on guesses:
            self._status_dset_name = 'completed_guess_positions'
            # Note that check_for_duplicates() will be against fit's parm_dict. So make a backup of that
            fit_parms = self.parms_dict.copy()
            # Set parms_dict to an empty dict so that we can accept any Guess dataset:
            self.parms_dict = dict()
            guess_complete_h5_grps, guess_partial_h5_grps = self._check_for_duplicates()
            if self.verbose and self.mpi_rank == 0:
                print(
                    'Guess datasets search resulted in:\nCompleted: {}\nPartial:{}'.format(
                        guess_complete_h5_grps, guess_partial_h5_grps))
            # Now put back the original parms_dict:
            self.parms_dict.update(fit_parms)

            # Case 2.1: At least guess is completed:
            if len(guess_complete_h5_grps) > 0:
                # Just set the last group as the current results group
                self.h5_results_grp = guess_complete_h5_grps[-1]
                if self.verbose and self.mpi_rank == 0:
                    print('Guess found! Using Guess in:\n{}'.format(
                        self.h5_results_grp))
                # It will grab the older status default unless we set the status dataset back to fit
                self._status_dset_name = 'completed_fit_positions'
                # Get handles to the guess dataset. Nothing else will be found
                self._get_existing_datasets()

            elif len(guess_complete_h5_grps) == 0 and len(
                    guess_partial_h5_grps) > 0:
                FileNotFoundError(
                    'Guess not yet completed. Please complete guess first')
                return
            else:
                FileNotFoundError(
                    'No Guess found. Please complete guess first')
                return

        # We want compute to call our own manual unit computation function:
        self._unit_computation = self._unit_compute_fit
