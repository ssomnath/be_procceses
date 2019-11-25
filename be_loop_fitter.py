# -*- coding: utf-8 -*-
"""
:class:`~pycroscopy.analysis.be_loop_fitter.BELoopFitter` that fits Simple Harmonic Oscillator model data to a
parametric model to describe hysteretic switching in ferroelectric materials

Created on Thu Aug 25 11:48:53 2016

@author: Suhas Somnath, Chris R. Smith, Rama K. Vasudevan

"""

from __future__ import division, print_function, absolute_import, unicode_literals
from enum import Enum
import joblib
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import least_squares
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from be_loop import projectLoop, fit_loop, generate_guess, loop_fit_function, calc_switching_coef_vec, switching32
from tree import ClusterTree
from be_sho_fitter import sho32

from pyUSID.io.dtype_utils import stack_real_to_compound
from pyUSID.io.hdf_utils import get_unit_values, get_sort_order, \
    reshape_to_n_dims, get_attr, create_empty_dataset, create_results_group, \
    write_reduced_anc_dsets, write_simple_attrs, write_main_dataset
from pyUSID.processing.comp_utils import get_MPI, recommend_cpu_cores
from pyUSID.io.usi_data import USIDataset

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

        self.process_name = "Loop_Fit"
        self.parms_dict = None

        self._check_validity(h5_main)

        if 'DC_Offset' in self.h5_main.spec_dim_labels:
            self._fit_dim_name = 'DC_Offset'
        elif 'write_bias' in self.h5_main.spec_dim_labels:
            self._fit_dim_name = 'write_bias'
        else:
            raise ValueError('Neither "DC_Offset", nor "write_bias" were '
                             'spectroscopic dimension in the provided dataset '
                             'which has dimensions: {}'
                             '.'.format(self.h5_main.spec_dim_labels))

        if 'FORC' in self.h5_main.spec_dim_labels:
            self._forc_dim_name = 'FORC'
        else:
            self._forc_dim_name = 'FORC_Cycle'

        # TODO: Need to catch KeyError s that would be thrown when attempting to access attributes
        file_data_type = get_attr(h5_main.file, 'data_type')
        meas_grp_name = h5_main.name.split('/')
        h5_meas_grp = h5_main.file[meas_grp_name[1]]
        meas_data_type = get_attr(h5_meas_grp, 'data_type')

        if h5_main.dtype != sho32:
            raise TypeError('Provided dataset is not a SHO results dataset.')

        # This check is clunky but should account for case differences.
        # If Python2 support is dropped, simplify with# single check using case
        if not (
                meas_data_type.lower != file_data_type.lower or meas_data_type.upper != file_data_type.upper):
            message = 'Mismatch between file and Measurement group data types for the chosen dataset.\n'
            message += 'File data type is {}.  The data type for Measurement group {} is {}'.format(
                file_data_type,
                h5_meas_grp.name,
                meas_data_type)
            raise ValueError(message)

        if file_data_type == 'BEPSData':
            if get_attr(h5_meas_grp, 'VS_mode') not in ['DC modulation mode',
                                                        'current mode']:
                raise ValueError(
                    'Provided dataset has a mode: "' + get_attr(h5_meas_grp,
                                                                'VS_mode') + '" is not a '
                                                                             '"DC modulation" or "current mode" BEPS dataset')
            elif get_attr(h5_meas_grp, 'VS_cycle_fraction') != 'full':
                raise ValueError('Provided dataset does not have full cycles')

        elif file_data_type == 'cKPFMData':
            if get_attr(h5_meas_grp, 'VS_mode') != 'cKPFM':
                raise ValueError(
                    'Provided dataset has an unsupported VS_mode: "' + get_attr(
                        h5_meas_grp, 'VS_mode') + '"')

        # #####################################################################
        # accounting for memory copies
        self._max_raw_pos_per_read = self._max_pos_per_read

        # TODO: oset limits in the set up functions
        self.results_pix_byte_size = (loop_fit32.itemsize + loop_metrics32.itemsize) * 64

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

    def __create_projection_datasets(self):
        """
        Setup the Loop_Fit Group and the loop projection datasets

        """

        # Which row in the spec datasets is DC offset?
        _fit_spec_index = self.h5_main.spec_dim_labels.index(
            self._fit_dim_name)

        # TODO: Unkown usage of variable. Waste either way
        # self._fit_offset_index = 1 + _fit_spec_index

        # Calculate the number of loops per position
        cycle_start_inds = np.argwhere(
            self.h5_main.h5_spec_inds[_fit_spec_index, :] == 0).flatten()
        tot_cycles = cycle_start_inds.size
        if self.verbose:
            print('Found {} cycles starting at indices: {}'.format(tot_cycles,
                                                                   cycle_start_inds))

        # Make the results group
        self.h5_results_grp = create_results_group(self.h5_main,
                                                   self.process_name)
        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        # Write datasets
        self.h5_projected_loops = create_empty_dataset(self.h5_main,
                                                       np.float32,
                                                       'Projected_Loops',
                                                       h5_group=self.h5_results_grp)

        h5_loop_met_spec_inds, h5_loop_met_spec_vals = write_reduced_anc_dsets(
            self.h5_results_grp, self.h5_main.h5_spec_inds,
            self.h5_main.h5_spec_vals, self._fit_dim_name,
            basename='Loop_Metrics', verbose=False)

        self.h5_loop_metrics = write_main_dataset(self.h5_results_grp, (
        self.h5_main.shape[0], tot_cycles), 'Loop_Metrics',
                                                  'Metrics', 'compound', None,
                                                  None, dtype=loop_metrics32,
                                                  h5_pos_inds=self.h5_main.h5_pos_inds,
                                                  h5_pos_vals=self.h5_main.h5_pos_vals,
                                                  h5_spec_inds=h5_loop_met_spec_inds,
                                                  h5_spec_vals=h5_loop_met_spec_vals)

        # Copy region reference:
        # copy_region_refs(self.h5_main, self.h5_projected_loops)
        # copy_region_refs(self.h5_main, self.h5_loop_metrics)

        self.h5_main.file.flush()
        self._met_spec_inds = self.h5_loop_metrics.h5_spec_inds

        if self.verbose and self.mpi_rank == 0:
            print('Finished creating Guess dataset')

    def _create_guess_datasets(self):
        """
        Creates the HDF5 Guess dataset
        """
        self.__create_projection_datasets()

        self.h5_guess = create_empty_dataset(self.h5_loop_metrics, loop_fit32,
                                             'Guess')

        self.h5_guess = USIDataset(self.h5_guess)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        self.h5_main.file.flush()

    def _create_fit_datasets(self):
        """
        Creates the HDF5 Fit dataset
        """

        if self.h5_guess is None:
            raise ValueError('Need to guess before fitting!')

        self.h5_fit = create_empty_dataset(self.h5_guess, loop_fit32, 'Fit')
        self.h5_fit = USIDataset(self.h5_fit)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        self.h5_main.file.flush()

    def _read_data_chunk(self):
        """
        Get the next chunk of raw data for doing the loop projections.
        """

        # The Process class should take care of all the basic reading
        super(BELoopFitter, self)._read_data_chunk()

        if self.data is None:
            # Nothing we can do at this point
            return

        if self.verbose and self.mpi_rank == 0:
            print('BELoopFitter got raw data of shape {} from super'
                  '.'.format(self.data.shape))

        """
        Now self.data contains data for N pixels. 
        The challenge is that this may contain M FORC cycles 
        Each FORC cycle needs its own V DC vector
        So, we can't blindly use the inherited unit_compute. 
        Our variables now are Position, Vdc, FORC, all others

        We want M lists of [VDC x all other variables]

        The challenge is that VDC and FORC are inner dimensions - 
        neither the fastest nor the slowest (guaranteed)
        """
        spec_dim_order_s2f = get_sort_order(self.h5_main.h5_spec_inds)[::-1]

        # order_to_s2f = list(pos_dim_order_s2f) + list( len(pos_dim_order_s2f) + spec_dim_order_s2f)
        self._dim_labels_s2f = list(['Positions']) + list(
            np.array(self.h5_main.spec_dim_labels)[spec_dim_order_s2f])

        self._num_forcs = int(
            any([targ in self.h5_main.spec_dim_labels for targ in
                 ['FORC', 'FORC_Cycle']]))

        order_to_s2f = [0] + list(1 + spec_dim_order_s2f)
        if self.verbose and self.mpi_rank == 0:
            print('Order for reshaping to S2F: {}'.format(order_to_s2f))

        if self.verbose and self.mpi_rank == 0:
            print(self._dim_labels_s2f, order_to_s2f)

        if self._num_forcs:
            forc_pos = self.h5_main.spec_dim_labels.index(self._forc_dim_name)
            self._num_forcs = self.h5_main.spec_dim_sizes[forc_pos]

        if self.verbose and self.mpi_rank == 0:
            print('Num FORCS: {}'.format(self._num_forcs))

        all_but_forc_rows = []
        for ind, dim_name in enumerate(self.h5_main.spec_dim_labels):
            if dim_name not in ['FORC', 'FORC_Cycle', 'FORC_repeat']:
                all_but_forc_rows.append(ind)

        if self.verbose and self.mpi_rank == 0:
            print('All but FORC rows: {}'.format(all_but_forc_rows))

        dc_mats = []

        forc_mats = []

        num_reps = 1 if self._num_forcs == 0 else self._num_forcs
        for forc_ind in range(num_reps):
            if self.verbose and self.mpi_rank == 0:
                print('\nWorking on FORC #{}'.format(forc_ind))

            if self._num_forcs:
                this_forc_spec_inds = \
                    np.where(self.h5_main.h5_spec_inds[forc_pos] == forc_ind)[
                        0]
            else:
                this_forc_spec_inds = np.ones(
                    shape=self.h5_main.h5_spec_inds.shape[1], dtype=np.bool)

            if self.verbose and self.mpi_rank == 0:
                print('Spectroscopic slice: {}'.format(this_forc_spec_inds))

            if self._num_forcs:
                this_forc_dc_vec = get_unit_values(
                    self.h5_main.h5_spec_inds[all_but_forc_rows][:,
                    this_forc_spec_inds],
                    self.h5_main.h5_spec_vals[all_but_forc_rows][:,
                    this_forc_spec_inds],
                    all_dim_names=list(np.array(self.h5_main.spec_dim_labels)[
                                           all_but_forc_rows]),
                    dim_names=self._fit_dim_name)
            else:
                this_forc_dc_vec = get_unit_values(self.h5_main.h5_spec_inds,
                                                   self.h5_main.h5_spec_vals,
                                                   dim_names=self._fit_dim_name)
            this_forc_dc_vec = this_forc_dc_vec[self._fit_dim_name]
            dc_mats.append(this_forc_dc_vec)

            this_forc_2d = self.data[:, this_forc_spec_inds]
            if self.verbose and self.mpi_rank == 0:
                print('2D slice shape for this FORC: {}'.format(this_forc_2d.shape))

            this_forc_nd, success = reshape_to_n_dims(this_forc_2d,
                                                      h5_pos=None,
                                                      h5_spec=self.h5_main.h5_spec_inds[
                                                              :,
                                                              this_forc_spec_inds])
            if self.verbose and self.mpi_rank == 0:
                print(this_forc_nd.shape)

            this_forc_nd_s2f = this_forc_nd.transpose(
                order_to_s2f).squeeze()  # squeeze out FORC
            dim_names_s2f = self._dim_labels_s2f.copy()
            if self._num_forcs > 0:
                dim_names_s2f.remove(
                    self._forc_dim_name)  # because it was never there in the first place.
            if self.verbose and self.mpi_rank == 0:
                print('Reordered to S2F: {}, {}'.format(this_forc_nd_s2f.shape,
                                                    dim_names_s2f))

            rest_dc_order = list(range(len(dim_names_s2f)))
            _dc_ind = dim_names_s2f.index(self._fit_dim_name)
            rest_dc_order.remove(_dc_ind)
            rest_dc_order = rest_dc_order + [_dc_ind]
            if self.verbose and self.mpi_rank == 0:
                print('Transpose for reordering to rest, DC: {}'.format(
                rest_dc_order))

            rest_dc_nd = this_forc_nd_s2f.transpose(rest_dc_order)
            rest_dc_names = list(np.array(dim_names_s2f)[rest_dc_order])

            self._pre_flattening_shape = list(rest_dc_nd.shape)
            self._pre_flattening_dim_name_order = list(rest_dc_names)

            if self.verbose and self.mpi_rank == 0:
                print('After reodering: {}, {}'.format(rest_dc_nd.shape,
                                                       rest_dc_names))

            dc_rest_2d = rest_dc_nd.reshape(np.prod(rest_dc_nd.shape[:-1]),
                                            np.prod(rest_dc_nd.shape[-1]))

            if self.verbose and self.mpi_rank == 0:
                print('Shape after flattening to 2D: {}'.format(dc_rest_2d.shape))

            forc_mats.append(dc_rest_2d)

        self.data = forc_mats, dc_mats

        if self.verbose and self.mpi_rank == 0:
            print('self.data loaded')

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
        super(BELoopFitter, self)._read_guess_chunk()

        print('h5_guess is:')
        print(self.h5_guess)

        spec_dim_order_s2f = get_sort_order(self.h5_guess.h5_spec_inds)[::-1]

        order_to_s2f = [0] + list(1 + spec_dim_order_s2f)
        print('Order for reshaping to S2F: {}'.format(order_to_s2f))

        dim_labels_s2f = list(['Positions']) + list(
            np.array(self.h5_guess.spec_dim_labels)[spec_dim_order_s2f])

        print(dim_labels_s2f, order_to_s2f)

        num_forcs = int(any([targ in self.h5_guess.spec_dim_labels for targ in
                             ['FORC', 'FORC_Cycle']]))
        if num_forcs:
            forc_pos = self.h5_guess.spec_dim_labels.index(self._forc_dim_name)
            num_forcs = self.h5_guess.spec_dim_sizes[forc_pos]
        print('Num FORCS: {}'.format(num_forcs))

        all_but_forc_rows = []
        for ind, dim_name in enumerate(self.h5_guess.spec_dim_labels):
            if dim_name not in ['FORC', 'FORC_Cycle', 'FORC_repeat']:
                all_but_forc_rows.append(ind)
        print('All but FORC rows: {}'.format(all_but_forc_rows))

        forc_mats = []

        num_reps = 1 if num_forcs == 0 else num_forcs
        for forc_ind in range(num_reps):
            print('')
            print('Working on FORC #{}'.format(forc_ind))
            if num_forcs:
                this_forc_spec_inds = \
                np.where(self.h5_guess.h5_spec_inds[forc_pos] == forc_ind)[0]
            else:
                this_forc_spec_inds = np.ones(
                    shape=self.h5_guess.h5_spec_inds.shape[1], dtype=np.bool)

            this_forc_2d = self.guess[:, this_forc_spec_inds]
            print('2D slice shape for this FORC: {}'.format(this_forc_2d.shape))

            this_forc_nd, success = reshape_to_n_dims(this_forc_2d,
                                                      h5_pos=None,
                                                      h5_spec=self.h5_guess.h5_spec_inds[
                                                              :,
                                                              this_forc_spec_inds])
            print(this_forc_nd.shape)

            this_forc_nd_s2f = this_forc_nd.transpose(
                order_to_s2f).squeeze()  # squeeze out FORC
            dim_names_s2f = dim_labels_s2f.copy()
            if num_forcs > 0:
                dim_names_s2f.remove(self._forc_dim_name)
                # because it was never there in the first place.
            print('Reordered to S2F: {}, {}'.format(this_forc_nd_s2f.shape,
                                                    dim_names_s2f))

            dc_rest_2d = this_forc_nd_s2f.ravel()
            print('Shape after flattening to 2D: {}'.format(dc_rest_2d.shape))
            forc_mats.append(dc_rest_2d)

        self.guess = forc_mats

    @staticmethod
    def _project_loop(sho_response, dc_offset):
        # projected_loop = np.zeros(shape=sho_response.shape, dtype=np.float32)
        ancillary = np.zeros(shape=1, dtype=loop_metrics32)

        pix_dict = projectLoop(np.squeeze(dc_offset),
                               sho_response['Amplitude [V]'],
                               sho_response['Phase [rad]'])

        projected_loop = pix_dict['Projected Loop']
        ancillary['Rotation Angle [rad]'] = pix_dict['Rotation Matrix'][0]
        ancillary['Offset'] = pix_dict['Rotation Matrix'][1]
        ancillary['Area'] = pix_dict['Geometric Area']
        ancillary['Centroid x'] = pix_dict['Centroid'][0]
        ancillary['Centroid y'] = pix_dict['Centroid'][1]

        return projected_loop, ancillary

    @staticmethod
    def __compute_batches(resp_2d_list, dc_vec_list, map_func, req_cores,
                          verbose=False):

        if verbose:
            print('Unit computation found {} FORC datasets with {} corresponding DC vectors'.format(len(resp_2d_list), len(dc_vec_list)))
            print('First dataset of shape: {}'.format(resp_2d_list[0].shape))

        MPI = get_MPI()
        if MPI is not None:
            rank = MPI.COMM_WORLD.Get_rank()
            cores = 1
        else:
            rank = 0
            cores = req_cores

        if verbose:
            print(
                'Rank {} starting loop projections on {} cores (requested {} cores)'.format(
                    rank, cores, req_cores))

        if cores > 1:
            values = []
            for loops_2d, curr_vdc in zip(resp_2d_list, dc_vec_list):
                print(loops_2d.shape, curr_vdc.shape, map_func)
                values += [joblib.delayed(map_func)(x, [curr_vdc])
                           for x
                           in loops_2d]
            results = joblib.Parallel(n_jobs=cores)(values)

            # Finished reading the entire data set
            print('Rank {} finished parallel computation'.format(rank))

        else:
            if verbose:
                print("Rank {} computing serially ...".format(rank))
            # List comprehension vs map vs for loop?
            # https://stackoverflow.com/questions/1247486/python-list-comprehension-vs-map
            results = []
            for loops_2d, curr_vdc in zip(resp_2d_list, dc_vec_list):
                results += [map_func(vector, curr_vdc) for vector in
                            loops_2d]

        return results

    @staticmethod
    def _guess_loops(vdc_vec, projected_loops_2d):
        """
        Provides loop parameter guesses for a given set of loops

        Parameters
        ----------
        vdc_vec : 1D numpy float numpy array
            DC voltage offsets for the loops
        projected_loops_2d : 2D numpy float array
            Projected loops arranged as [instance or position x dc voltage steps]

        Returns
        -------
        guess_parms : 1D compound numpy array
            Loop parameter guesses for the provided projected loops

        """

        def _loop_fit_tree(tree, guess_mat, fit_results, vdc_shifted,
                           shift_ind):
            """
            Recursive function that fits a tree object describing the cluster results

            Parameters
            ----------
            tree : ClusterTree object
                Tree describing the clustering results
            guess_mat : 1D numpy float array
                Loop parameters that serve as guesses for the loops in the tree
            fit_results : 1D numpy float array
                Loop parameters that serve as fits for the loops in the tree
            vdc_shifted : 1D numpy float array
                DC voltages shifted be 1/4 cycle
            shift_ind : unsigned int
                Number of units to shift loops by

            Returns
            -------
            guess_mat : 1D numpy float array
                Loop parameters that serve as guesses for the loops in the tree
            fit_results : 1D numpy float array
                Loop parameters that serve as fits for the loops in the tree

            """
            # print('Now fitting cluster #{}'.format(tree.name))
            # I already have a guess. Now fit myself
            curr_fit_results = fit_loop(vdc_shifted,
                                        np.roll(tree.value, shift_ind),
                                        guess_mat[tree.name])
            # keep all the fit results
            fit_results[tree.name] = curr_fit_results
            for child in tree.children:
                # Use my fit as a guess for the lower layers:
                guess_mat[child.name] = curr_fit_results[0].x
                # Fit this child:
                guess_mat, fit_mat = _loop_fit_tree(child, guess_mat,
                                                    fit_results, vdc_shifted,
                                                    shift_ind)
            return guess_mat, fit_results

        num_clusters = max(2, int(projected_loops_2d.shape[
                                      0] ** 0.5))  # change this to 0.6 if necessary
        estimators = KMeans(num_clusters)
        results = estimators.fit(projected_loops_2d)
        centroids = results.cluster_centers_
        labels = results.labels_

        # Get the distance between cluster means
        distance_mat = pdist(centroids)
        # get hierarchical pairings of clusters
        linkage_pairing = linkage(distance_mat, 'weighted')
        # Normalize the pairwise distance with the maximum distance
        linkage_pairing[:, 2] = linkage_pairing[:, 2] / max(
            linkage_pairing[:, 2])

        # Now use the tree class:
        cluster_tree = ClusterTree(linkage_pairing[:, :2], labels,
                                   distances=linkage_pairing[:, 2],
                                   centroids=centroids)
        num_nodes = len(cluster_tree.nodes)

        # prepare the guess and fit matrices
        loop_guess_mat = np.zeros(shape=(num_nodes, 9), dtype=np.float32)
        # loop_fit_mat = np.zeros(shape=loop_guess_mat.shape, dtype=loop_guess_mat.dtype)
        loop_fit_results = list(
            np.arange(num_nodes, dtype=np.uint16))  # temporary placeholder

        shift_ind, vdc_shifted = BELoopFitter.shift_vdc(vdc_vec)

        # guess the top (or last) node
        loop_guess_mat[-1] = generate_guess(vdc_vec, cluster_tree.tree.value)

        # Now guess the rest of the tree
        loop_guess_mat, loop_fit_results = _loop_fit_tree(cluster_tree.tree,
                                                          loop_guess_mat,
                                                          loop_fit_results,
                                                          vdc_shifted,
                                                          shift_ind)

        # Prepare guesses for each pixel using the fit of the cluster it belongs to:
        guess_parms = np.zeros(shape=projected_loops_2d.shape[0],
                               dtype=loop_fit32)
        for clust_id in range(num_clusters):
            pix_inds = np.where(labels == clust_id)[0]
            temp = np.atleast_2d(loop_fit_results[clust_id][0].x)
            # convert to the appropriate dtype as well:
            r2 = 1 - np.sum(np.abs(loop_fit_results[clust_id][0].fun ** 2))
            guess_parms[pix_inds] = stack_real_to_compound(
                np.hstack([temp, np.atleast_2d(r2)]), loop_fit32)

        return guess_parms

    @staticmethod
    def shift_vdc(vdc_vec):
        """
        Rolls the Vdc vector by a quarter cycle

        Parameters
        ----------
        vdc_vec : 1D numpy array
            DC offset vector

        Returns
        -------
        shift_ind : int
            Number of indices by which the vector was rolled
        vdc_shifted : 1D numpy array
            Vdc vector rolled by a quarter cycle

        """
        shift_ind = int(
            -1 * len(vdc_vec) / 4)  # should NOT be hardcoded like this!
        vdc_shifted = np.roll(vdc_vec, shift_ind)
        return shift_ind, vdc_shifted

    def __guess_chunk(self, *args, **kwargs):
        if self.verbose and self.mpi_rank == 0:
            print("Rank {} at custom _unit_computation".format(self.mpi_rank))

        resp_2d_list, dc_vec_list = self.data

        results = self.__compute_batches(resp_2d_list, dc_vec_list, self._project_loop, self._cores, verbose=self.verbose)

        # Step 1: unzip the two components in results into separate arrays
        if self.verbose and self.mpi_rank == 0:
            print('Unzipping loop projection results')
        loop_mets = np.zeros(shape=len(results), dtype=loop_metrics32)
        proj_loops = np.zeros(shape=(len(results), self.data[0][0].shape[1]),
                              dtype=np.float32)

        if self.verbose and self.mpi_rank == 0:
            print(
                'Prepared empty arrays for loop metrics of shape: {} and '
                'projected loops of shape: {}.'
                ''.format(loop_mets.shape, proj_loops.shape))

        for ind in range(len(results)):
            proj_loops[ind] = results[ind][0]
            loop_mets[ind] = results[ind][1]

        # NOW do the guess:
        proj_forc = proj_loops.reshape((len(dc_vec_list),
                                        len(results) // len(dc_vec_list),
                                        proj_loops.shape[-1]))

        if self.verbose and self.mpi_rank == 0:
            print('Reshaped projected loops from {} to: {}'.format(
                proj_loops.shape, proj_forc.shape))

        # Convert forc dimension to a list
        if self.verbose and self.mpi_rank == 0:
            print('Going to compute guesses now')

        all_guesses = []

        for proj_loops_this_forc, curr_vdc in zip(proj_forc, dc_vec_list):
            # this works on batches and not individual loops
            # Cannot be done in parallel
            this_guesses = self._guess_loops(curr_vdc, proj_loops_this_forc)
            all_guesses.append(this_guesses)

        self._results = proj_loops, loop_mets, np.array(all_guesses)

    def set_up_guess(self, h5_partial_guess=None):
        self.parms_dict = {'projection_method': 'pycroscopy BE loop model',
                           'guess_method': "pycroscopy Cluster Tree"}

        # ask super to take care of the rest, which is a standardized operation
        super(BELoopFitter, self).set_up_guess(h5_partial_guess=h5_partial_guess)

        self._max_pos_per_read = self._max_raw_pos_per_read // 1.5

        self._unit_computation = self.__guess_chunk
        self.compute = self.do_guess
        self._write_results_chunk = self._write_guess_chunk

    def set_up_fit(self, h5_partial_fit=None, h5_guess=None, ):
        self.parms_dict = {'fit_method': 'pycroscopy functional'}

        # ask super to take care of the rest, which is a standardized operation
        super(BELoopFitter, self).set_up_fit(h5_partial_fit=h5_partial_fit,
                                             h5_guess=h5_guess)

        self._max_pos_per_read = self._max_raw_pos_per_read // 1.5

        self._unit_computation = self._unit_compute_fit
        self.compute = self.do_fit
        self._write_results_chunk = self._write_fit_chunk

        print('Status dataset name is: ' + self._status_dset_name)
        print([item for item in self.h5_results_grp])

    @staticmethod
    def BE_LOOP(coef_vec, data_vec, dc_vec, *args):
        """

        Parameters
        ----------
        coef_vec : numpy.ndarray
        data_vec : numpy.ndarray
        dc_vec : numpy.ndarray
            The DC offset vector
        args : list

        Returns
        -------
        fitness : float
            The 1-r^2 value for the current set of loop coefficients

        """

        if coef_vec.size < 9:
            raise ValueError(
                'Error: The Loop Fit requires 9 parameter guesses!')

        data_mean = np.mean(data_vec)

        func = loop_fit_function(dc_vec, coef_vec)

        ss_tot = sum(abs(data_vec - data_mean) ** 2)
        ss_res = sum(abs(data_vec - func) ** 2)

        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return 1 - r_squared

    def _unit_compute_fit(self, *args, **kwargs):

        # 1 - r_squared = sho_error(guess, data_vec, freq_vector)

        obj_func = self.BE_LOOP
        solver_options = {'jac': 'cs'}

        resp_2d_list, dc_vec_list = self.data

        # At this point data has been read in. Read in the guess as well:
        self._read_guess_chunk()

        if self.verbose and self.mpi_rank == 0:
            print('_unit_compute_fit got:\nobj_func: {}\n'
                  'solver_options: {}'.format(obj_func, solver_options))

        # TODO: Generalize this bit. Use Parallel compute instead!

        if self.mpi_size > 1:
            if self.verbose:
                print('Rank {}: About to start serial computation'
                      '.'.format(self.mpi_rank))

            self._results = list()
            for dc_vec, loops_2d, guess_parms in zip(dc_vec_list, resp_2d_list, self.guess):
                '''
                Shift the loops and vdc vector
                '''
                shift_ind, vdc_shifted = self.shift_vdc(dc_vec)
                loops_2d_shifted = np.roll(loops_2d, shift_ind, axis=0).T

                for loop_resp, loop_guess in zip(loops_2d_shifted, guess_parms):
                    curr_results = least_squares(obj_func, loop_guess,
                                                 args=[loop_resp, dc_vec],
                                                 **solver_options)
                    self._results.append(curr_results)
        else:
            cores = recommend_cpu_cores(len(resp_2d_list) * resp_2d_list[0].shape[0],
                                        verbose=self.verbose)
            if self.verbose:
                print('Starting parallel fitting with {} cores'.format(cores))

            values = list()
            for dc_vec, loops_2d, guess_parms in zip(dc_vec_list, resp_2d_list, self.guess):
                temp = [joblib.delayed(least_squares)(obj_func, this_guess,
                                                    args=[this_loop, dc_vec],
                                                    **solver_options) for
                      this_loop, this_guess in zip(loops_2d, guess_parms)]
                values.append(temp)
            self._results = joblib.Parallel(n_jobs=cores)(values)

        if self.verbose and self.mpi_rank == 0:
            print(
                'Finished computing fits on {} objects. Results of length: {}'
                '.'.format(self.data.shape[0], len(self._results)))

        # What least_squares returns is an object that needs to be extracted
        # to get the coefficients. This is handled by the write function

    @staticmethod
    def _reformat_results_chunk(num_forcs, proj_loops, first_n_dim_shape,
                                first_n_dim_names, dim_labels_s2f,
                                forc_dim_name, verbose=False):

        # What we need to do is put the forc back as the slowest dimension before the pre_flattening shape:
        if num_forcs > 1:
            first_n_dim_shape = [num_forcs] + first_n_dim_shape
            first_n_dim_names = [forc_dim_name] + first_n_dim_names
        if verbose:
            print('Dimension sizes & order: {} and names: {} that flattened '
                  'results will be reshaped to'
                  '.'.format(first_n_dim_shape, first_n_dim_names))

        # Now, reshape the flattened 2D results to its N-dim form before flattening (now FORC included):
        first_n_dim_results = proj_loops.reshape(first_n_dim_shape)

        # Need to put data back to slowest >> fastest dim
        map_to_s2f = [first_n_dim_names.index(dim_name) for dim_name in
                      dim_labels_s2f]
        if verbose:
            print('Will permute as: {} to arrange dimensions from slowest to '
                  'fastest varying'.format(map_to_s2f))

        results_nd_s2f = first_n_dim_results.transpose(map_to_s2f)

        if verbose:
            print('Shape: {} and dimension labels: {} of results arranged from'
                  ' slowest to fastest varying'
                  '.'.format(results_nd_s2f.shape, dim_labels_s2f))

        pos_size = np.prod(results_nd_s2f.shape[:1])
        spec_size = np.prod(results_nd_s2f.shape[1:])

        if verbose:
            print('Results will be flattend to: {}'
                  '.'.format((pos_size, spec_size)))

        results_2d = results_nd_s2f.reshape(pos_size, spec_size)

        return results_2d

    def _write_guess_chunk(self):

        """
        self._results is now a zipped tuple containing:
        1. a projected loop (an array of float32) and
        2. a single compound element for hte loop metrics

        Step 1 will be to unzip the two components into separate arrays
        Step 2 will fold back the flattened 1 / 2D array into the N-dim form
        Step 3 will reverse all transposes
        Step 4 will flatten back to its original 2D form
        Step 5 will finally write the data to an HDF5 file
        """

        proj_loops, loop_mets, all_guesses = self._results

        if self.verbose:
            print('Unzipped results into Projected loops and Metrics arrays')

        # Step 2: Fold to N-D before reversing transposes:
        loops_2d = self._reformat_results_chunk(self._num_forcs, proj_loops,
                                                self._pre_flattening_shape,
                                                self._pre_flattening_dim_name_order,
                                                self._dim_labels_s2f,
                                                self._forc_dim_name,
                                                verbose=self.verbose)

        met_labels_s2f = self._dim_labels_s2f.copy()
        met_labels_s2f.remove(self._fit_dim_name)

        mets_2d = self._reformat_results_chunk(self._num_forcs, loop_mets,
                                               self._pre_flattening_shape[:-1],
                                               self._pre_flattening_dim_name_order[:-1],
                                               met_labels_s2f,
                                               self._forc_dim_name,
                                               verbose=self.verbose)

        guess_2d = self._reformat_results_chunk(self._num_forcs, all_guesses,
                                               self._pre_flattening_shape[:-1],
                                               self._pre_flattening_dim_name_order[:-1],
                                               met_labels_s2f,
                                               self._forc_dim_name,
                                               verbose=self.verbose)

        # Which pixels are we working on?
        curr_pixels = self._get_pixels_in_current_batch()

        if self.verbose:
            print(
                'Writing projected loops of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(
                    loops_2d.shape, loops_2d.dtype,
                    self.h5_projected_loops.shape,
                    self.h5_projected_loops.dtype))
            print(
                'Writing loop metrics of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(
                    mets_2d.shape, mets_2d.dtype, self.h5_loop_metrics.shape,
                    self.h5_loop_metrics.dtype))

            print(
                'Writing Guesses of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(
                    guess_2d.shape, guess_2d.dtype, self.h5_guess.shape,
                    self.h5_guess.dtype))

        self.h5_projected_loops[curr_pixels, :] = loops_2d
        self.h5_loop_metrics[curr_pixels, :] = mets_2d
        self.h5_guess[curr_pixels, :] = guess_2d

    def _write_fit_chunk(self):

        """
        self._results is now a zipped tuple containing:
        1. a projected loop (an array of float32) and
        2. a single compound element for hte loop metrics

        Step 1 will be to unzip the two components into separate arrays
        Step 2 will fold back the flattened 1 / 2D array into the N-dim form
        Step 3 will reverse all transposes
        Step 4 will flatten back to its original 2D form
        Step 5 will finally write the data to an HDF5 file
        """

        all_fits = np.array(self._results)

        if self.verbose:
            print('Unzipped results into Projected loops and Metrics arrays')

        met_labels_s2f = self._dim_labels_s2f.copy()
        met_labels_s2f.remove(self._fit_dim_name)

        fits_2d = self._reformat_results_chunk(self._num_forcs, all_fits,
                                               self._pre_flattening_shape[:-1],
                                               self._pre_flattening_dim_name_order[:-1],
                                               met_labels_s2f,
                                               self._forc_dim_name,
                                               verbose=self.verbose)

        # Which pixels are we working on?
        curr_pixels = self._get_pixels_in_current_batch()

        if self.verbose:
            print(
                'Writing Fits of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(
                    fits_2d.shape, fits_2d.dtype, self.h5_fit.shape,
                    self.h5_fit.dtype))

        self.h5_fits[curr_pixels, :] = fits_2d
