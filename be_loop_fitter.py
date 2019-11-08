# -*- coding: utf-8 -*-
"""
:class:`~pycroscopy.analysis.be_loop_fitter.BELoopFitter` that fits Simple Harmonic Oscillator model data to a
parametric model to describe hysteretic switching in ferroelectric materials

Created on Thu Aug 25 11:48:53 2016

@author: Suhas Somnath, Chris R. Smith, Rama K. Vasudevan

"""

from __future__ import division, print_function, absolute_import, unicode_literals
from enum import Enum
from warnings import warn
import numpy as np
import scipy
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from be_loop import projectLoop, fit_loop, generate_guess, calc_switching_coef_vec, switching32
from tree import ClusterTree
from be_sho_fitter import sho32

from pyUSID.io.dtype_utils import flatten_compound_to_real, stack_real_to_compound
from pyUSID.io.hdf_utils import copy_region_refs, \
    get_sort_order, get_dimensionality, reshape_to_n_dims, reshape_from_n_dims, get_attr, \
    create_empty_dataset, create_results_group, write_reduced_spec_dsets, write_simple_attrs, write_main_dataset
from pyUSID.processing.comp_utils import get_available_memory
from pyUSID import USIDataset

from fitter import Fitter

'''
Custom dtypes for the datasets created during fitting.
'''
loop_metrics32 = np.dtype({'names': ['Area', 'Centroid x', 'Centroid y', 'Rotation Angle [rad]', 'Offset'],
                           'formats': [np.float32, np.float32, np.float32, np.float32, np.float32]})

crit32 = np.dtype({'names': ['AIC_loop', 'BIC_loop', 'AIC_line', 'BIC_line'],
                   'formats': [np.float32, np.float32, np.float32, np.float32]})

field_names = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'b_0', 'b_1', 'b_2', 'b_3', 'R2 Criterion']
loop_fit32 = np.dtype({'names': field_names,
                       'formats': [np.float32 for name in field_names]})


class LoopGuessFunc(Enum):
    cluster = 0


class LoopFitFunc(Enum):
    least_squares = 0


class BELoopFitter(Fitter):
    """
    A class that fits Simple Harmonic Oscillator model data to a 9-parameter model to describe hysteretic switching in
    ferroelectric materials

    Parameters
    ----------
    h5_main : h5py.Dataset instance
        The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
        indices and values, and position indices and values datasets.
    variables : list(string), Default ['Frequency']
        Lists of attributes that h5_main should possess so that it may be analyzed by Model.
    parallel : bool, optional
        Should the parallel implementation of the fitting be used.  Default True.

    Returns
    -------
    None

    Notes
    -----
    Quantitative mapping of switching behavior in piezoresponse force microscopy, Stephen Jesse, Ho Nyung Lee,
    and Sergei V. Kalinin, Review of Scientific Instruments 77, 073702 (2006); doi: http://dx.doi.org/10.1063/1.2214699

    """

    def __init__(self, h5_main, variables=None, **kwargs):
        if variables is None:
            variables = ['DC_Offset']

        super(BELoopFitter, self).__init__(h5_main, variables=variables, **kwargs)
        self._check_validity(h5_main)

        self.process_name = "Loop_Fit"
        self.parms_dict = None

        self.h5_guess_parameters = None
        self.h5_fit_parameters = None
        self._sho_spec_inds = None
        self._sho_spec_vals = None  # used only at one location. can remove if deemed unnecessary
        self._met_spec_inds = None
        self._num_forcs = 1
        self._num_forc_repeats = 1
        self._sho_pos_inds = None
        self._current_pos_slice = slice(None)
        self._current_sho_spec_slice = slice(None)
        self._current_met_spec_slice = slice(None)
        self._fit_offset_index = 0
        self._sho_all_but_forc_inds = None
        self._sho_all_but_dc_forc_inds = None
        self._met_all_but_forc_inds = None
        self._current_forc = 0
        self._maxDataChunk = 1
        self._fit_dim_name = variables[0]

    @staticmethod
    def _check_validity(h5_main):
        """
        Checks whether or not the provided object can be analyzed by this class.

        Parameters
        ----------
        h5_main : h5py.Dataset instance
            The dataset containing the SHO Fit (not necessarily the dataset directly resulting from SHO fit)
            over which the loop projection, guess, and fit will be performed.
        """
        # TODO: Need to catch KeyError s that would be thrown when attempting to access attributes
        file_data_type = get_attr(h5_main.file, 'data_type')
        meas_grp_name = h5_main.name.split('/')
        h5_meas_grp = h5_main.file[meas_grp_name[1]]
        meas_data_type = get_attr(h5_meas_grp, 'data_type')

        if h5_main.dtype != sho32:
            raise TypeError('Provided dataset is not a SHO results dataset.')

        # This check is clunky but should account for case differences.  If Python2 support is dropped, simplify with
        # single check using casefold.
        if not (meas_data_type.lower != file_data_type.lower or meas_data_type.upper != file_data_type.upper):
            message = 'Mismatch between file and Measurement group data types for the chosen dataset.\n'
            message += 'File data type is {}.  The data type for Measurement group {} is {}'.format(file_data_type,
                                                                                               h5_meas_grp.name,
                                                                                               meas_data_type)
            raise ValueError(message)

        if file_data_type == 'BEPSData':
            if get_attr(h5_meas_grp, 'VS_mode') not in ['DC modulation mode', 'current mode']:
                raise ValueError('Provided dataset has a mode: "' + get_attr(h5_meas_grp, 'VS_mode') + '" is not a '
                                 '"DC modulation" or "current mode" BEPS dataset')
            elif get_attr(h5_meas_grp, 'VS_cycle_fraction') != 'full':
                raise ValueError('Provided dataset does not have full cycles')

        elif file_data_type == 'cKPFMData':
            if get_attr(h5_meas_grp, 'VS_mode') != 'cKPFM':
                raise ValueError('Provided dataset has an unsupported VS_mode: "' + get_attr(h5_meas_grp, 'VS_mode') + '"')

    def _create_projection_datasets(self):
        """
        Setup the Loop_Fit Group and the loop projection datasets

        """
        # First grab the spectroscopic indices and values and position indices
        self._sho_spec_inds = self.h5_main.h5_spec_inds
        self._sho_spec_vals = self.h5_main.h5_spec_vals
        self._sho_pos_inds = self.h5_main.h5_pos_inds

        fit_dim_ind = self.h5_main.spec_dim_labels.index(self._fit_dim_name)

        self._fit_spec_index = fit_dim_ind
        self._fit_offset_index = 1 + fit_dim_ind

        # Calculate the number of loops per position
        cycle_start_inds = np.argwhere(self._sho_spec_inds[fit_dim_ind, :] == 0).flatten()
        tot_cycles = cycle_start_inds.size

        # Make the results group
        self.h5_results_grp = create_results_group(self.h5_main, self.process_name)
        write_simple_attrs(self.h5_results_grp, {'projection_method': 'pycroscopy BE loop model'})

        # Write datasets
        self.h5_projected_loops = create_empty_dataset(self.h5_main, np.float32, 'Projected_Loops',
                                                       h5_group=self.h5_results_grp)

        h5_loop_met_spec_inds, h5_loop_met_spec_vals = write_reduced_spec_dsets(self.h5_results_grp, self._sho_spec_inds,
                                                                                self._sho_spec_vals, self._fit_dim_name,
                                                                                basename='Loop_Metrics')

        self.h5_loop_metrics = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], tot_cycles), 'Loop_Metrics',
                                                  'Metrics', 'compound', None, None, dtype=loop_metrics32,
                                                  h5_pos_inds=self.h5_main.h5_pos_inds,
                                                  h5_pos_vals=self.h5_main.h5_pos_vals,
                                                  h5_spec_inds=h5_loop_met_spec_inds,
                                                  h5_spec_vals=h5_loop_met_spec_vals)

        # Copy region reference:
        copy_region_refs(self.h5_main, self.h5_projected_loops)
        copy_region_refs(self.h5_main, self.h5_loop_metrics)

        self.h5_main.file.flush()
        self._met_spec_inds = self.h5_loop_metrics.h5_spec_inds

    def _reshape_projected_loops_for_h5(self, projected_loops_2d, order_dc_offset_reverse,
                                        nd_mat_shape_dc_first):
        """
        Reshapes the 2D projected loops to the format such that they can be written to the h5 file

        Parameters
        ----------
        projected_loops_2d : 2D numpy float array
            Projected loops arranged as [instance or position x dc voltage steps]
        order_dc_offset_reverse : tuple of unsigned ints
            Order in which the N dimensional data should be transposed to return it to the format used in h5 files
        nd_mat_shape_dc_first : 1D numpy unsigned int array
            Shape of the N dimensional array that the loops_2d can be turned into.
            We use the order_dc_offset_reverse after this reshape

        Returns
        -------
        proj_loops_2d : 2D numpy float array
            Projected loops reshaped to the original chronological order in which the data was acquired
        """
        if self.verbose:
            print('Projected loops of shape:', projected_loops_2d.shape, ', need to bring to:', nd_mat_shape_dc_first)
        # Step 9: Reshape back to same shape as fit_Nd2:
        projected_loops_nd = np.reshape(projected_loops_2d, nd_mat_shape_dc_first)
        if self.verbose:
            print('Projected loops reshaped to N dimensions :', projected_loops_nd.shape)
        # Step 10: Move Vdc back inwards. Only for projected loop
        projected_loops_nd_2 = np.transpose(projected_loops_nd, order_dc_offset_reverse)
        if self.verbose:
            print('Projected loops after moving DC offset inwards:', projected_loops_nd_2.shape)
        # step 11: reshape back to 2D
        proj_loops_2d, success = reshape_from_n_dims(projected_loops_nd_2,
                                                     h5_pos=None,
                                                     h5_spec=self._sho_spec_inds[self._sho_all_but_forc_inds,
                                                                                 self._current_sho_spec_slice])
        if not success:
            warn('unable to reshape projected loops')
            return None
        if self.verbose:
            print('loops shape after collapsing dimensions:', proj_loops_2d.shape)

        return proj_loops_2d

    @staticmethod
    def _project_loop_batch(dc_offset, sho_mat):
        """
        This function projects loops given a matrix of the amplitude and phase.
        These matrices (and the Vdc vector) must have a single cycle's worth of
        points on the second dimension

        Parameters
        ----------
        dc_offset : 1D list or numpy array
            DC voltages. vector of length N
        sho_mat : 2D compound numpy array of type - sho32
            SHO response matrix of size MxN - [pixel, dc voltage]

        Returns
        -------
        results : tuple
            Results from projecting the provided matrices with following components

            projected_loop_mat : MxN numpy array
                Array of Projected loops
            ancillary_mat : M, compound numpy array
                This matrix contains the ancillary information extracted when projecting the loop.
                It contains the following components per loop:
                    'Area' : geometric area of the loop

                    'Centroid x': x positions of centroids for each projected loop

                    'Centroid y': y positions of centroids for each projected loop

                    'Rotation Angle': Angle by which loop was rotated [rad]

                    'Offset': Offset removed from loop
        Note
        -----
        This is the function that can be made parallel if need be.
        However, it is fast enough as is
        """
        num_pixels = int(sho_mat.shape[0])
        projected_loop_mat = np.zeros(shape=sho_mat.shape, dtype=np.float32)
        ancillary_mat = np.zeros(shape=num_pixels, dtype=loop_metrics32)

        for pixel in range(num_pixels):
            """if pixel % 50 == 0:
                print("Projecting Loop {} of {}".format(pixel, num_pixels))"""

            pix_dict = projectLoop(np.squeeze(dc_offset),
                                   sho_mat[pixel]['Amplitude [V]'],
                                   sho_mat[pixel]['Phase [rad]'])

            projected_loop_mat[pixel, :] = pix_dict['Projected Loop']
            ancillary_mat[pixel]['Rotation Angle [rad]'] = pix_dict['Rotation Matrix'][0]
            ancillary_mat[pixel]['Offset'] = pix_dict['Rotation Matrix'][1]
            ancillary_mat[pixel]['Area'] = pix_dict['Geometric Area']
            ancillary_mat[pixel]['Centroid x'] = pix_dict['Centroid'][0]
            ancillary_mat[pixel]['Centroid y'] = pix_dict['Centroid'][1]

        return projected_loop_mat, ancillary_mat

    def _get_sho_chunk_sizes(self, max_mem_mb):
        """
        Calculates the largest number of positions that can be read into memory for a single FORC cycle

        Parameters
        ----------
        max_mem_mb : unsigned int
            Maximum allowable memory in megabytes
        verbose : Boolean (Optional. Default is False)
            Whether or not to print debugging statements

        Returns
        -------
        max_pos : unsigned int
            largest number of positions that can be read into memory for a single FORC cycle
        sho_spec_inds_per_forc : unsigned int
            Number of indices in the SHO spectroscopic table that will be used per read
        metrics_spec_inds_per_forc : unsigned int
            Number of indices in the Loop metrics spectroscopic table that will be used per read
        """
        # Step 1: Find number of FORC cycles and repeats (if any), DC steps, and number of loops
        # dc_offset_index = np.argwhere(self._sho_spec_inds.attrs['labels'] == 'DC_Offset').squeeze()
        num_dc_steps = np.unique(self._sho_spec_inds[self._fit_spec_index, :]).size
        all_spec_dims = list(range(self._sho_spec_inds.shape[0]))
        all_spec_dims.remove(self._fit_spec_index)

        # Remove FORC_cycles
        sho_spec_labels = self.h5_main.spec_dim_labels
        has_forcs = 'FORC' in sho_spec_labels or 'FORC_Cycle' in sho_spec_labels
        if has_forcs:
            forc_name = 'FORC' if 'FORC' in sho_spec_labels else 'FORC_Cycle'
            try:
                forc_pos = sho_spec_labels.index(forc_name)
            except Exception:
                raise
            # forc_pos = np.argwhere(sho_spec_labels == forc_name)[0][0]
            self._num_forcs = np.unique(self._sho_spec_inds[forc_pos]).size
            all_spec_dims.remove(forc_pos)

            # Remove FORC_repeats
            has_forc_repeats = 'FORC_repeat' in sho_spec_labels
            if has_forc_repeats:
                try:
                    forc_repeat_pos = sho_spec_labels.index('FORC_repeat')
                except Exception:
                    raise
                # forc_repeat_pos = np.argwhere(sho_spec_labels == 'FORC_repeat')[0][0]
                self._num_forc_repeats = np.unique(self._sho_spec_inds[forc_repeat_pos]).size
                all_spec_dims.remove(forc_repeat_pos)

        # calculate number of loops:
        if len(all_spec_dims) == 0:
            loop_dims = 1
        else:
            # Now calculate number of repetitions in all dimensions besides FORC and DC offset:
            loop_dims = get_dimensionality(self._sho_spec_inds[all_spec_dims, :])
        loops_per_forc = np.product(loop_dims)

        # Step 2: Calculate the largest number of FORCS and positions that can be read given memory limits:
        size_per_forc = num_dc_steps * loops_per_forc * len(self.h5_main.dtype) * self.h5_main.dtype[0].itemsize
        """
        How we arrive at the number for the overhead (how many times the size of the data-chunk we will use in memory)
        1 for the original data, 1 for data copied to all children processes, 1 for results, 0.5 for fit, guess, misc
        """
        mem_overhead = 3.5
        max_pos = int(max_mem_mb * 1024 ** 2 / (size_per_forc * mem_overhead))
        if self.verbose:
            print('Can read {} of {} pixels given a {} MB memory limit'.format(max_pos,
                                                                               self._sho_pos_inds.shape[0],
                                                                               max_mem_mb))
        self.max_pos = int(min(self._sho_pos_inds.shape[0], max_pos))
        self.sho_spec_inds_per_forc = int(self._sho_spec_inds.shape[1] / self._num_forcs / self._num_forc_repeats)
        self.metrics_spec_inds_per_forc = int(self._met_spec_inds.shape[1] / self._num_forcs / self._num_forc_repeats)

        # Step 3: Read allowed chunk
        self._sho_all_but_forc_inds = list(range(self._sho_spec_inds.shape[0]))
        self._met_all_but_forc_inds = list(range(self._met_spec_inds.shape[0]))
        if self._num_forcs > 1:
            self._sho_all_but_forc_inds.remove(forc_pos)
            met_forc_pos = np.argwhere(get_attr(self._met_spec_inds, 'labels') == forc_name)[0][0]
            self._met_all_but_forc_inds.remove(met_forc_pos)

            if self._num_forc_repeats > 1:
                self._sho_all_but_forc_inds.remove(forc_repeat_pos)
                met_forc_repeat_pos = np.argwhere(get_attr(self._met_spec_inds, 'labels') == 'FORC_repeat')[0][0]
                self._met_all_but_forc_inds.remove(met_forc_repeat_pos)
                
    def _get_dc_offset(self):
        """
        Gets the DC offset for the current FORC step

        Parameters
        ----------
        verbose : boolean (optional)
            Whether or not to print debugging statements

        Returns
        -------
        dc_vec : 1D float numpy array
            DC offsets for the current FORC step
        """
        # apply this knowledge to reshape the spectroscopic values
        # remember to reshape such that the dimensions are arranged in reverse order (slow to fast)
        spec_vals_nd, success = reshape_to_n_dims(self._sho_spec_vals[self._sho_all_but_forc_inds,
                                                                      self._current_sho_spec_slice],
                                                  h5_spec=self._sho_spec_inds[self._sho_all_but_forc_inds,
                                                                              self._current_sho_spec_slice])
        # This should result in a N+1 dimensional matrix where the first index contains the actual data
        # the other dimensions are present to easily slice the data
        spec_labels_sorted = np.hstack(('Dim', self.h5_main.spec_dim_labels))
        if self.verbose:
            print('Spectroscopic dimensions sorted by rate of change:')
            print(spec_labels_sorted)
        # slice the N dimensional dataset such that we only get the DC offset for default values of other dims
        fit_dim_pos = np.argwhere(spec_labels_sorted == self._fit_dim_name)[0][0]
        # fit_dim_slice = list()
        # for dim_ind in range(spec_labels_sorted.size):
        #     if dim_ind == fit_dim_pos:
        #         fit_dim_slice.append(slice(None))
        #     else:
        #         fit_dim_slice.append(slice(0, 1))

        fit_dim_slice = [fit_dim_pos]
        for idim, dim in enumerate(spec_labels_sorted[1:]):
            if dim == self._fit_dim_name:
                fit_dim_slice.append(slice(None))
                fit_dim_slice[0] = idim
            elif dim in ['FORC', 'FORC_repeat', 'FORC_Cycle']:
                continue
            else:
                fit_dim_slice.append(slice(0, 1))

        if self.verbose:
            print('slice to extract Vdc:')
            print(fit_dim_slice)

        self.fit_dim_vec = np.squeeze(spec_vals_nd[tuple(fit_dim_slice)])

        return

    def _project_loops(self):
        """
        Do the projection of the SHO fit
        """

        self._create_projection_datasets()
        self._get_sho_chunk_sizes(0.25 * get_available_memory())

        '''
        Loop over the FORCs
        '''
        for forc_chunk_index in range(self._num_forcs):
            pos_chunk_index = 0

            self._current_sho_spec_slice = slice(self.sho_spec_inds_per_forc * self._current_forc,
                                                 self.sho_spec_inds_per_forc * (self._current_forc + 1))
            self._current_met_spec_slice = slice(self.metrics_spec_inds_per_forc * self._current_forc,
                                                 self.metrics_spec_inds_per_forc * (self._current_forc + 1))
            dc_vec = self._get_dc_offset()
            '''
            Loop over positions
            '''

            self._end_pos =
            while self._current_pos_slice.stop < self._end_pos:
                loops_2d, nd_mat_shape_dc_first, order_dc_offset_reverse = self._get_projection_data(pos_chunk_index)

                # step 8: perform loop unfolding
                projected_loops_2d, loop_metrics_1d = self._project_loop_batch(dc_vec, np.transpose(loops_2d))

                # test the reshapes back
                projected_loops_2d = self._reshape_projected_loops_for_h5(projected_loops_2d,
                                                                          order_dc_offset_reverse,
                                                                          nd_mat_shape_dc_first)
                self.h5_projected_loops[self._current_pos_slice, self._current_sho_spec_slice] = projected_loops_2d

                metrics_2d = self._reshape_results_for_h5(loop_metrics_1d, nd_mat_shape_dc_first)

                self.h5_loop_metrics[self._current_pos_slice, self._current_met_spec_slice] = metrics_2d

            # Reset the position slice
            self._current_pos_slice = slice(None)

    def _create_guess_datasets(self):
        """
        Creates the HDF5 Guess dataset and links the it to the ancillary datasets.
        """
        self.h5_guess = create_empty_dataset(self.h5_loop_metrics, loop_fit32, 'Guess')
        write_simple_attrs(self._h5_group, {'guess method': 'pycroscopy statistical'})

        # This is necessary comparing against new runs to avoid re-computation + resuming partial computation
        write_simple_attrs(self.h5_guess, self._parms_dict)
        write_simple_attrs(self.h5_guess, {'Loop_fit_method': "pycroscopy statistical", 'last_pixel': 0})

        self.h5_main.file.flush()

    def _create_fit_datasets(self):
        pass

    def _read_data_chunk(self):
        pass

    def _read_guess_chunk(self):
        pass

    def _write_results_chunk(self):
        pass

    def set_up_guess(self, *func_args, h5_partial_guess=None, **func_kwargs):
        pass

    def set_up_fit(self, *func_args, h5_partial_fit=None, h5_guess=None, **func_kwargs):
        pass

    def _unit_compute_fit(self, *args, **kwargs):
        pass

