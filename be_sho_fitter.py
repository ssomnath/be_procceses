import sys
from enum import Enum
from warnings import warn
import numpy as np
import joblib
from functools import partial
from scipy.optimize import least_squares

sys.path.append(r'C:\Users\Suhas\PycharmProjects\pyUSID')
from pyUSID import USIDataset
from pyUSID.io.hdf_utils import copy_region_refs, write_simple_attrs, create_results_group, write_reduced_spec_dsets, \
                                create_empty_dataset, write_main_dataset
from pyUSID.processing.comp_utils import recommend_cpu_cores

# From this project:
from utils import *
from fitter import Fitter


'''
Custom dtype for the datasets created during fitting.
'''
field_names = ['Amplitude [V]', 'Frequency [Hz]', 'Quality Factor', 'Phase [rad]', 'R2 Criterion']
sho32 = np.dtype({'names': field_names,
                  'formats': [np.float32 for name in field_names]})


class SHOGuessFunc(Enum):
    complex_gaussian = 0
    wavelet_peaks = 1


class SHOFitFunc(Enum):
    least_squares = 0


class BESHOfitter(Fitter):
    
    def __init__(self, h5_main, **kwargs):
        super(BESHOfitter, self).__init__(h5_main, variables=['Frequency'], **kwargs)

        self.process_name = "SHO_Fit"
        self.parms_dict = None
        
        self._fit_dim_name = 'Frequency'      

        # Extract some basic parameters that are necessary for either the guess or fit
        freq_dim_ind = self.h5_main.spec_dim_labels.index('Frequency')
        self.step_start_inds = np.where(self.h5_main.h5_spec_inds[freq_dim_ind] == 0)[0]
        self.num_udvs_steps = len(self.step_start_inds)

        # find the frequency vector and hold in memory
        self.freq_vec = None
        self._get_frequency_vector()

        # This is almost always True but think of this more as a sanity check.
        self.is_reshapable = is_reshapable(self.h5_main, self.step_start_inds)

        # accounting for memory copies
        self._max_raw_pos_per_read = self._max_pos_per_read
        # set limits in the set up functions
            
    def _get_frequency_vector(self):
        """
        Reads the frequency vector from the Spectroscopic_Values dataset.  
        This assumes that the data is reshape-able.
        
        """
        h5_spec_vals = self.h5_main.h5_spec_vals
        freq_dim = np.argwhere('Frequency' == np.array(self.h5_main.spec_dim_labels)).squeeze()

        if len(self.step_start_inds) == 1:  # BE-Line
            end_ind = h5_spec_vals.shape[1]
        else:  # BEPS
            end_ind = self.step_start_inds[1]

        self.freq_vec = h5_spec_vals[freq_dim, self.step_start_inds[0]:end_ind]
          
    def _create_guess_datasets(self):
        """
        Creates the h5 group, guess dataset, corresponding spectroscopic datasets and also
        links the guess dataset to the spectroscopic datasets.
        """
        self.h5_results_grp = create_results_group(self.h5_main, self.process_name)
        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        h5_sho_inds, h5_sho_vals = write_reduced_spec_dsets(self.h5_results_grp, self.h5_main.h5_spec_inds,
                                                            self.h5_main.h5_spec_vals, self._fit_dim_name)

        self.h5_guess = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], self.num_udvs_steps), 'Guess', 'SHO',
                                           'compound', None, None, h5_pos_inds=self.h5_main.h5_pos_inds,
                                           h5_pos_vals=self.h5_main.h5_pos_vals, h5_spec_inds=h5_sho_inds,
                                           h5_spec_vals=h5_sho_vals, chunks=(1, self.num_udvs_steps), dtype=sho32,
                                           main_dset_attrs=self.parms_dict, verbose=self.verbose)
        
        copy_region_refs(self.h5_main, self.h5_guess)
        
        self.h5_guess.file.flush()
        
        if self.verbose and self.mpi_rank == 0:
            print('Finished creating Guess dataset')
        
    def _create_fit_datasets(self):
        """
        Creates the HDF5 fit dataset. pycroscopy requires that the h5 group, guess dataset,
        corresponding spectroscopic and position datasets be created and populated at this point.
        This function will create the HDF5 dataset for the fit and link it to same ancillary datasets as the guess.
        The fit dataset will NOT be populated here but will instead be populated using the __setData function
        """

        if self.h5_guess is None or self.h5_results_grp is None:
            warn('Need to guess before fitting!')
            return

        """
        Once the guess is complete, the last_pixel attribute will be set to complete for the group.
        Once the fit is initiated, during the creation of the status dataset, this last_pixel
        attribute will be used and it wil make the fit look like it was already complete. Which is not the case.
        This is a problem of doing two processes within the same group. 
        Until all legacy is removed, we will simply reset the last_pixel attribute.
        """
        self.h5_results_grp.attrs['last_pixel'] = 0

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        # Create the fit dataset as an empty dataset of the same size and dtype as the guess.
        # Also automatically links in the ancillary datasets.
        self.h5_fit = USIDataset(create_empty_dataset(self.h5_guess, dtype=sho32, dset_name='Fit'))

        self.h5_fit.file.flush()
        
        if self.verbose and self.mpi_rank == 0:
            print('Finished creating Fit dataset')
        
    def _read_data_chunk(self):
        """
        Returns the next chunk of data for the guess or the fit
        """

        # The Fitter class should take care of all the basic reading
        super(BESHOfitter, self)._read_data_chunk()

        # At this point the self.data object is the raw data that needs to be reshaped to a single UDVS step:
        if self.data is not None:
            if self.verbose and self.mpi_rank == 0:
                print('Got raw data of shape {} from super'.format(self.data.shape))
            self.data = reshape_to_one_step(self.data, self.num_udvs_steps)
            if self.verbose and self.mpi_rank == 0:
                print('Reshaped raw data to shape {}'.format(self.data.shape))
                
    def _read_guess_chunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset.

        Parameters
        -----
        None

        Returns
        --------

        """
        # The Fitter class should take care of all the basic reading
        super(BESHOfitter, self)._read_guess_chunk()
        
        self.guess = reshape_to_one_step(self.guess, self.num_udvs_steps)
        # bear in mind that this self.guess is a compound dataset. Convert to float32
        # don't keep the R^2.
        self.guess = np.hstack([self.guess[name] for name in self.guess.dtype.names if name != 'R2 Criterion'])
                    
    def _write_results_chunk(self):
        """
        Writes the provided chunk of data into the guess or fit datasets. 
        This method is responsible for any and all book-keeping.
        """
        prefix = 'guess' if self._is_guess else 'fit'
        self._results = self._reformat_results(self._results,
                                               self.parms_dict[prefix + '-algorithm'])
        
        if self._is_guess:
            self.guess = np.hstack(tuple(self._results))
            # prepare to reshape:
            self.guess = np.transpose(np.atleast_2d(self.guess))
            if self.verbose and self.mpi_rank == 0:
                print('Prepared guess of shape {} before reshaping'.format(self.guess.shape))
            self.guess = reshape_to_n_steps(self.guess, self.num_udvs_steps)
            if self.verbose and self.mpi_rank == 0:
                print('Reshaped guess to shape {}'.format(self.guess.shape))
        else:
            self.fit = self._results
            self.fit = np.transpose(np.atleast_2d(self.fit))
            self.fit = reshape_to_n_steps(self.fit, self.num_udvs_steps)

        # ask super to take care of the rest, which is a standardized operation
        super(BESHOfitter, self)._write_results_chunk()
                   
    def set_up_guess(self, guess_func=SHOGuessFunc.complex_gaussian, 
                     *func_args, h5_partial_guess=None, **func_kwargs):
        """
        Need this because during the set up, we won't know which strategy is being used.
        Should Guess be its own Process class in that case? If so, it would end up having 
        its own group etc.
        
        Move generic code to Fitter
        """
        self.parms_dict = {'guess-method': "pycroscopy BESHO"}
        
        if not isinstance(guess_func, SHOGuessFunc):
            raise TypeError('Please supply SHOGuessFunc.complex_gaussian or SHOGuessFunc.wavelet_peaks for the guess_func')
        
        partial_func = None
        
        if guess_func == SHOGuessFunc.complex_gaussian:
            
            num_points=func_kwargs.pop('num_points', 5)
            
            self.parms_dict.update({'guess-algorithm': 'complex_gaussian',
                                    'guess-complex_gaussian-num_points': num_points})
            
            partial_func = partial(complex_gaussian, w_vec=self.freq_vec, 
                                   num_points=num_points)
            
        elif guess_func == SHOGuessFunc.wavelet_peaks:
            
            peak_width_bounds = func_kwargs.pop('peak_width_bounds', [10, 200])
            peak_width_step = func_kwargs.pop('peak_width_step', 20)

            if len(func_args) > 0:
                # Assume that the first argument is what we are looking for
                peak_width_bounds = func_args[0]
            
            self.parms_dict.update({'guess_algorithm': 'wavelet_peaks',
                                    'guess-wavelet_peaks-peak_width_bounds': peak_width_bounds,
                                    'guess-wavelet_peaks-peak_width_step': peak_width_step})

            partial_func = partial(wavelet_peaks, peak_width_bounds=peak_width_bounds, 
                                   peak_width_step=peak_width_step, **func_kwargs)
                        
        # Assuming that Guess has not taken place: 
        # Set up the parms dict so everything necessary for checking previous guess / fit is ready
        self._is_guess = True
        self._status_dset_name = 'completed_guess_positions'
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()
        
        if self.verbose and self.mpi_rank == 0:
               print('Groups with Guess in:\nCompleted: {}\nPartial:{}'.format(self.duplicate_h5_groups, self.partial_h5_groups))
        
        self._map_function = partial_func
        self._unit_computation = super(BESHOfitter, self)._unit_computation
        self._create_results_datasets = self._create_guess_datasets
        self._max_pos_per_read = 25 # self._max_raw_pos_per_read // 1.2
        
    def set_up_fit(self, fit_func=SHOFitFunc.least_squares, 
                   *func_args, h5_partial_fit=None, h5_guess=None, **func_kwargs):
        """
        Need this because during the set up, we won't know which strategy is being used.
        Should Guess be its own Process class in that case? If so, it would end up having 
        its own group etc.
        """
        self.parms_dict = {'fit-method': "pycroscopy BESHO"}
        
        if not isinstance(fit_func, SHOFitFunc):
            raise TypeError('Please supply SHOFitFunc.least_squares for the fit_func')
                
        if fit_func == SHOFitFunc.least_squares:
                                    
            self.parms_dict.update({'fit-algorithm': 'least_squares'})
             
        self._is_guess = False
        
        self._map_function = None
        self._unit_computation = None
        self._create_results_datasets = self._create_fit_datasets
        
        # Case 1: Fit already complete or partially complete. This is similar to a partial process. Leave as is
        self._status_dset_name = 'completed_fit_positions'
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()
        if self.verbose and self.mpi_rank == 0:
            print('Checking on partial / completed fit datasets')
            print('Completed results groups:\n{}\nPartial results groups:\n{}'.format(self.duplicate_h5_groups,
                                                                                      self.partial_h5_groups))

        # Case 2: Fit neither partial / completed. Search for guess. Most popular scenario:
        if len(self.duplicate_h5_groups) == 0 and len(self.partial_h5_groups) == 0:
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
                print('Guess datasets search resulted in:\nCompleted: {}\nPartial:{}'.format(guess_complete_h5_grps, guess_partial_h5_grps))
            # Now put back the original parms_dict:
            self.parms_dict.update(fit_parms)
            
            # Case 2.1: At least guess is completed:
            if len(guess_complete_h5_grps) > 0:
                # Just set the last group as the current results group
                self.h5_results_grp = guess_complete_h5_grps[-1]
                if self.verbose and self.mpi_rank == 0:
                    print('Guess found! Using Guess in:\n{}'.format(self.h5_results_grp))
                # It will grab the older status default unless we set the status dataset back to fit
                self._status_dset_name = 'completed_fit_positions'
                # Get handles to the guess dataset. Nothing else will be found 
                self._get_existing_datasets()
                
            elif len(guess_complete_h5_grps) == 0 and len(guess_partial_h5_grps) > 0:
                FileNotFoundError('Guess not yet completed. Please complete guess first')
                return
            else:
                FileNotFoundError('No Guess found. Please complete guess first')
                return
            
        if self.verbose and self.mpi_rank == 0:
            print('Name of status dataset: ' + self._status_dset_name)
            print('Parameters dictionary: {}'.format(self.parms_dict))
            print('Current results dataset: {}'.format(self.h5_results_grp))

        # We want compute to call our own manual unit computation function:
        self._unit_computation = self._unit_compute_fit
        self._max_pos_per_read = 25 # self._max_raw_pos_per_read // 1.4
           
    def _unit_compute_fit(self, *args, **kwargs):
        # At this point data has been read in. Read in the guess as well:
        self._read_guess_chunk()
        # Call joblib directly now and then parallel compute manually:
        
        # result = least_squares(sho_error, guess_parms, args=(resp_vec, freq_vec))
        """
        least_squares(fun, x0, jac='2-point', bounds=(-inf, inf), method='trf', ftol=1e-08, 
                      xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, 
                      tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={}
        """
        solver_options={'jac': 'cs'}
        
        values = [joblib.delayed(least_squares)(sho_error, pix_guess,
                                         args=[pulse_resp, self.freq_vec],
                                         **solver_options) for pulse_resp, pix_guess in zip(self.data, self.guess)]
        cores = recommend_cpu_cores(self.data.shape[0], verbose=self.verbose)
        self._results = joblib.Parallel(n_jobs=cores)(values)
        
        if self.verbose and self.mpi_rank == 0:
            print('Finished computing fits on {} spectras. Results currently of length: {}'.format(self.data.shape[0], len(self._results)))
                  
        # What least_squares returns is an object that needs to be extracted
        # to get the coefficients. This is handled by the write function
        
    def _reformat_results(self, results, strategy='wavelet_peaks'):
        """
        Model specific calculation and or reformatting of the raw guess or fit results

        Parameters
        ----------
        results : array-like
            Results to be formatted for writing
        strategy : str
            The strategy used in the fit.  Determines how the results will be reformatted.
            Default 'wavelet_peaks'

        Returns
        -------
        sho_vec : numpy.ndarray
            The reformatted array of parameters.
            
        """
        if self.verbose and self.mpi_rank == 0:
            print('Strategy to use for reformatting results: "{}"'.format(strategy))
        # Create an empty array to store the guess parameters
        sho_vec = np.zeros(shape=(len(results)), dtype=sho32)
        if self.verbose and self.mpi_rank == 0:
            print('Raw results and compound SHO vector of shape {}'.format(len(results)))

        # Extracting and reshaping the remaining parameters for SHO
        if strategy in ['wavelet_peaks', 'relative_maximum', 'absolute_maximum']:
            if self.verbose and self.mpi_rank == 0:
                  print('Reformatting results from a peak-position-finding algorithm')
            # wavelet_peaks sometimes finds 0, 1, 2, or more peaks. Need to handle that:
            # peak_inds = np.array([pixel[0] for pixel in results])
            peak_inds = np.zeros(shape=(len(results)), dtype=np.uint32)
            for pix_ind, pixel in enumerate(results):
                if len(pixel) == 1:  # majority of cases - one peak found
                    peak_inds[pix_ind] = pixel[0]
                elif len(pixel) == 0:  # no peak found
                    peak_inds[pix_ind] = int(0.5*self.data.shape[1])  # set to center of band
                else:  # more than one peak found
                    dist = np.abs(np.array(pixel) - int(0.5*self.data.shape[1]))
                    peak_inds[pix_ind] = pixel[np.argmin(dist)]  # set to peak closest to center of band
            if self.verbose and self.mpi_rank == 0:
                print('Peak positions of shape {}'.format(peak_inds.shape))
            # First get the value (from the raw data) at these positions:
            comp_vals = np.array(
                [self.data[pixel_ind, peak_inds[pixel_ind]] for pixel_ind in np.arange(peak_inds.size)])
            if self.verbose and self.mpi_rank == 0:
                print('Complex values at peak positions of shape {}'.format(comp_vals.shape))
            sho_vec['Amplitude [V]'] = np.abs(comp_vals)  # Amplitude
            sho_vec['Phase [rad]'] = np.angle(comp_vals)  # Phase in radians
            sho_vec['Frequency [Hz]'] = self.freq_vec[peak_inds]  # Frequency
            sho_vec['Quality Factor'] = np.ones_like(comp_vals) * 10  # Quality factor
            # Add something here for the R^2
            sho_vec['R2 Criterion'] = np.array([self.r_square(self.data, self._sho_func, self.freq_vec, sho_parms)
                                                for sho_parms in sho_vec])
        elif strategy in ['complex_gaussian']:
            if self.verbose and self.mpi_rank == 0:
                print('Reformatting results from the SHO Guess algorithm')
            for iresult, result in enumerate(results):
                sho_vec['Amplitude [V]'][iresult] = result[0]
                sho_vec['Frequency [Hz]'][iresult] = result[1]
                sho_vec['Quality Factor'][iresult] = result[2]
                sho_vec['Phase [rad]'][iresult] = result[3]
                sho_vec['R2 Criterion'][iresult] = result[4]
        elif strategy in ['least_squares']:
            if self.verbose and self.mpi_rank == 0:
                print('Reformatting results from a list of least_squares result objects')
            for iresult, result in enumerate(results):
                sho_vec['Amplitude [V]'][iresult] = result.x[0]
                sho_vec['Frequency [Hz]'][iresult] = result.x[1]
                sho_vec['Quality Factor'][iresult] = result.x[2]
                sho_vec['Phase [rad]'][iresult] = result.x[3]
                sho_vec['R2 Criterion'][iresult] = 1-result.fun
        else:
            if self.verbose and self.mpi_rank == 0:
                  print('_reformat_results() will not reformat results since the provided algorithm: {} does not match anything that this function can handle.'.format(strategy))

        return sho_vec
    