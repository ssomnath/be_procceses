import sys
from enum import Enum
from warnings import warn
import numpy as np
import joblib
from functools import partial
from scipy.optimize import least_squares

from pyUSID.io.hdf_utils import copy_region_refs, write_simple_attrs, create_results_group, write_reduced_anc_dsets, \
                                create_empty_dataset, write_main_dataset, get_attr, get_unit_values, reshape_to_n_dims, get_sort_order
from pyUSID.io.usi_data import USIDataset
from pyUSID.processing.process import Process
from pyUSID.processing.comp_utils import get_MPI

# From this project:
from be_sho_fitter import sho32
from be_loop import projectLoop
from fitter import Fitter


'''
Custom dtype for the datasets created during fitting.
'''

loop_metrics32 = np.dtype({'names': ['Area', 'Centroid x', 'Centroid y', 'Rotation Angle [rad]', 'Offset'],
                           'formats': [np.float32, np.float32, np.float32, np.float32, np.float32]})


class BELoopProjector(Process):
    
    def __init__(self, h5_main, **kwargs):
        super(BELoopProjector, self).__init__(h5_main, **kwargs)

        self._fit_dim_name = 'DC_Offset'

        if self._fit_dim_name not in self.h5_main.spec_dim_labels:
            raise ValueError('"' + self._fit_dim_name + '" not a spectroscopic dimension in the provided dataset which has dimensions: {}'.format(self.h5_main.spec_dim_labels))

        # TODO: Need to catch KeyError s that would be thrown when attempting to access attributes
        file_data_type = get_attr(h5_main.file, 'data_type')
        meas_grp_name = h5_main.name.split('/')
        h5_meas_grp = h5_main.file[meas_grp_name[1]]
        meas_data_type = get_attr(h5_meas_grp, 'data_type')

        if h5_main.dtype != sho32:
            raise TypeError('Provided dataset is not a SHO results dataset.')

        # This check is clunky but should account for case differences.  If Python2 support is dropped, simplify with
        # single check using casefold.
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

        self.process_name = "Loop_Projection"
        self.parms_dict = {'projection_method': 'pycroscopy BE loop model'}

        # Now Extract some basic parameters that are necessary for either the guess or fit
        self.dc_offsets_mat = self._get_dc_offsets(self.h5_main.h5_spec_inds,
                                                   self.h5_main.h5_spec_vals)

    def _create_projection_datasets(self):
        """
        Setup the Loop_Fit Group and the loop projection datasets

        """
        # First grab the spectroscopic indices and values and position indices
        # TODO: Avoid unnecessary namespace pollution
        # self._sho_spec_inds = self.h5_main.h5_spec_inds
        # self._sho_spec_vals = self.h5_main.h5_spec_vals
        # self._sho_pos_inds = self.h5_main.h5_pos_inds

        # Which row in the spec datasets is DC offset?
        self._fit_spec_index = self.h5_main.spec_dim_labels.index(self._fit_dim_name)

        # TODO: Unkown usage of variable. Waste either way
        # self._fit_offset_index = 1 + self._fit_spec_index

        # Calculate the number of loops per position
        cycle_start_inds = np.argwhere(self.h5_main.h5_spec_inds[self._fit_spec_index, :] == 0).flatten()
        tot_cycles = cycle_start_inds.size
        if self.verbose:
            print('Found {} cycles starting at indices: {}'.format(tot_cycles, cycle_start_inds))

        # Make the results group
        self.h5_results_grp = create_results_group(self.h5_main, self.process_name)
        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        # Write datasets
        self.h5_projected_loops = create_empty_dataset(self.h5_main, np.float32, 'Projected_Loops',
                                                       h5_group=self.h5_results_grp)

        h5_loop_met_spec_inds, h5_loop_met_spec_vals = write_reduced_anc_dsets(self.h5_results_grp, self.h5_main.h5_spec_inds,
                                                                                self.h5_main.h5_spec_vals, self._fit_dim_name,
                                                                                basename='Loop_Metrics', verbose=self.verbose)

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

        if self.verbose and self.mpi_rank == 0:
            print('Finished creating Guess dataset')
        
    def _read_data_chunk(self):
        """
        Returns the next chunk of data for the guess or the fit
        """

        # The Process class should take care of all the basic reading
        super(BELoopProjector, self)._read_data_chunk()

        if self.data is None:
            # Nothing we can do at this point
            return

        if self.verbose and self.mpi_rank == 0:
            print('BELoopProjector got raw data of shape {} from super'
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
        # resp_2d, dc_vec lists is what this function returns
        self.data = self.get_forc_pairs_from_sho_2d(self.data)
        if self.verbose and self.mpi_rank == 0:
            print('Reshaped raw data to {} FORC datasets, each of shape {}'
                  '.'.format(len(self.data[0]), self.data[0][0].shape))

    @staticmethod
    def _get_dc_offsets(h5_spec_inds, h5_spec_vals, verbose=False):
        # FORC is the decider whether or not DC_Offset changes.
        # FORC_Repeats etc. should not matter
        spec_unit_vals = get_unit_values(h5_spec_inds,
                                         h5_spec_vals,
                                         verbose=verbose)
        if 'FORC' not in spec_unit_vals.keys():
            if verbose:
                print(
                    'This is not a FORC dataset. Just taking unit values for DC Offset')
            dc_val_mat = np.expand_dims(spec_unit_vals['DC_Offset'], axis=0)
        else:
            # Reshape the Spec values matrix into an N dimensional array
            if verbose:
                print(
                    'This is a FORC dataset. Reshaping Spectroscopic Values to N dimensions')
            ret_vals = reshape_to_n_dims(h5_spec_vals,
                                         np.expand_dims(
                                             np.arange(h5_spec_vals.shape[0]),
                                             axis=1),
                                         h5_spec_inds, get_labels=True)
            spec_vals_nd, success, spec_nd_labels = ret_vals

            if success != True:
                raise ValueError(
                    'Unable to reshape Spectroscopic values to get DC offsets for each FORC')

            # We will be using "in" quite a bit. So convert to list
            spec_nd_labels = list(spec_nd_labels)

            if verbose:
                print('Reshaped Spectroscopic Values to: {}'.format(
                    spec_vals_nd.shape))
                print(
                    'Spectroscopic dimension names: {}'.format(spec_nd_labels))

            # Note the indices of all other dimensions
            all_other_dims = set(range(len(spec_nd_labels))) - \
                             set([spec_nd_labels.index('DC_Offset'),
                                  spec_nd_labels.index('FORC')])
            # Set up a new order where FORC is at 0 and DC is at 1 and all
            # other dimensions (useless) follow
            new_order = [spec_nd_labels.index('FORC'),
                         spec_nd_labels.index('DC_Offset')] + list(
                all_other_dims)
            if verbose:
                print('Will transpose this N-dim matrix as: {}'.format(
                    new_order))

            # Apply this new order to the matrix and the labels
            spec_vals_nd = spec_vals_nd.transpose(new_order)
            spec_nd_labels = np.array(spec_nd_labels)[new_order]
            if verbose:
                print('After transpose shape and names:\n\t{}\n\t{}'.format(
                    spec_vals_nd.shape, spec_nd_labels))

            # Now remove all other dimensions using a list of slices:
            keep_list = [slice(None), slice(None)] + [slice(0, 1) for _ in
                                                      range(
                                                          len(all_other_dims))]
            # Don't forget to remove singular dimensions using squeeze
            dc_val_mat = spec_vals_nd[keep_list].squeeze()
            # Unnecessary but let's keep track of dimension names anyway
            spec_nd_labels = spec_nd_labels[:2]
            if verbose:
                print(
                    'After removing all other dimensions. Shape is: {} and dimensions are: {}'.format(
                        dc_val_mat.shape, spec_nd_labels))

        return dc_val_mat

    @staticmethod
    def reshape_sho_chunk_to_nd(data_2d, raw_dim_labels,
                                h5_pos_inds, h5_spec_inds,
                                verbose=False):

        ret_vals = reshape_to_n_dims(data_2d, h5_pos_inds[:data_2d.shape[0]],
                                     h5_spec_inds)
        data_nd_auto, success = ret_vals

        if success != True:
            raise ValueError(
                'Unable to reshape data chunk of shape {} to N dimensions'.format(
                    data_2d.shape))

        if verbose:
            print('Reshaped raw data from: {} to {}'.format(data_2d.shape,
                                                            data_nd_auto.shape))

        # By default it is fast to slow!
        pos_sort = get_sort_order(h5_pos_inds)[::-1]
        spec_sort = get_sort_order(h5_spec_inds)[::-1]
        swap_order = list(pos_sort) + list(len(pos_sort) + spec_sort)

        if verbose:
            print(
                'Dimensions will be permuted as {} to arrange them from slowest to fastest'.format(
                    swap_order))

        data_nd_s2f = data_nd_auto.transpose(swap_order)
        dim_labels_s2f = np.array(raw_dim_labels)[swap_order]

        if verbose:
            print(
                'After rearranging array is of shape: {}, dimensions are ordered as: {}'.format(
                    data_nd_s2f.shape, dim_labels_s2f))

        return data_nd_s2f, dim_labels_s2f

    @staticmethod
    def break_nd_by_forc(data_nd_s2f, dim_labels_s2f, num_forcs,
                         verbose=False):

        if num_forcs > 1:
            # Fundamental assumption: FORC will always be the slowest dimension
            # YOu can have repeats, cycles etc. but all of those will
            # coreespond to the same FORC index - a single defintion for DC_Off
            forc_dim_ind = dim_labels_s2f.index('FORC')
            forc_less_labels_s2f = dim_labels_s2f[
                                   :forc_dim_ind] + dim_labels_s2f[
                                                    forc_dim_ind + 1:]
            if verbose:
                print(
                    '"FORC" was found at index: {} in the dimension labels (slow>>fast): {}'.format(
                        forc_dim_ind, dim_labels_s2f))
                print('Dimensions after removing FORC: {}'.format(
                    forc_less_labels_s2f))

            single_forc_indices = [slice(None) for _ in
                                   range(len(dim_labels_s2f))]
            forc_dsets = []
            switch = True
            for forc_ind in range(num_forcs):
                single_forc_indices[forc_dim_ind] = slice(forc_ind,
                                                          forc_ind + 1)
                temp = data_nd_s2f[single_forc_indices].squeeze()
                if verbose and switch:
                    print(
                        'Slice list used to slice index: {} of FORC: {}'.format(
                            forc_ind, single_forc_indices))
                    print('Shape of matrix after slicing FORC: {}'.format(
                        temp.shape))
                    switch = False

                forc_dsets.append(temp)
        else:
            forc_dsets = [data_nd_s2f]
            forc_less_labels_s2f = dim_labels_s2f

        return forc_dsets, list(forc_less_labels_s2f)

    @staticmethod
    def get_forc_pairs(forc_dsets, forc_less_labels_s2f, dc_val_mat,
                       verbose=False):

        dc_vec = []
        resp_2d = []

        switch = True
        for dc_offsets, forc_less_mat_nd in zip(dc_val_mat, forc_dsets):
            if len(forc_less_labels_s2f) != forc_less_mat_nd.ndim:
                raise ValueError('Length of labels: {} does not match with '
                                 'number of dimensions of dataset: {}'
                                 '.'.format(len(forc_less_labels_s2f),
                                            forc_less_mat_nd.ndim))

            dc_dim_ind = forc_less_labels_s2f.index('DC_Offset')
            if verbose and switch:
                print(
                    '"DC_Offset" found at index: {} in list of dimensions: {}'.format(
                        dc_dim_ind, forc_less_labels_s2f))

            slower_than_dc_dim_inds = list(range(dc_dim_ind))
            faster_than_dc_dim_inds = list(
                range(dc_dim_ind + 1, len(forc_less_labels_s2f)))
            trans_order = slower_than_dc_dim_inds + faster_than_dc_dim_inds + [
                dc_dim_ind]
            if verbose and switch:
                print(
                    'Transposing the data matrix as: {} to get "DC_Offset" as last dimension'.format(
                        trans_order))

            shifted_matrix = forc_less_mat_nd.transpose(trans_order)
            if verbose and switch:
                print(
                    'Shape of permuted array: {} with dimension names: {}'.format(
                        shifted_matrix.shape,
                        np.array(forc_less_labels_s2f)[trans_order]))

            all_x_vdc = shifted_matrix.reshape(-1, shifted_matrix.shape[-1])
            if verbose and switch:
                print('Shape of 2D array after flattening: {}'.format(
                    all_x_vdc.shape))
                switch = False

            dc_vec.append(dc_offsets)
            resp_2d.append(all_x_vdc)

        return resp_2d, dc_vec

    def get_forc_pairs_from_sho_2d(self, data_2d):

        data_nd_s2f, dim_labels_s2f = self.reshape_sho_chunk_to_nd(data_2d,
                                                                   self.h5_main.n_dim_labels,
                                                                   self.h5_main.h5_pos_inds,
                                                                   self.h5_main.h5_spec_inds,
                                                                   verbose=self.verbose)

        forc_dsets, forc_less_labels_s2f = self.break_nd_by_forc(data_nd_s2f,
                                                                 list(dim_labels_s2f),
                                                                 self.dc_offsets_mat.shape[0],
                                                                 verbose=self.verbose)

        return self.get_forc_pairs(forc_dsets, forc_less_labels_s2f,
                                   self.dc_offsets_mat, verbose=self.verbose)

    def _write_results_chunk(self):
        """
        Writes the provided chunk of data into the guess or fit datasets. 
        This method is responsible for any and all book-keeping.
        """
        pass

    def _unit_computation(self):
        if self.verbose and self.mpi_rank == 0:
            print("Rank {} at _unit_computation() that "
                  "will call parallel_compute()".format(self.mpi_rank))

        # resp_2d, dc_vec = self.data

        req_cores = self._cores
        MPI = get_MPI()
        if MPI is not None:
            rank = MPI.COMM_WORLD.Get_rank()
            cores = 1
        else:
            rank = 0
            cores = self._cores

        if self.verbose:
            print(
                'Rank {} starting computing on {} cores (requested {} cores)'.format(
                    rank, cores, req_cores))

        if cores > 1:
            values = []
            for loops_2d, curr_vdc in self.data:
                values += [joblib.delayed(self._map_function)(x, [curr_vdc]) for x
                          in loops_2d]
            results = joblib.Parallel(n_jobs=cores)(values)

            # Finished reading the entire data set
            print('Rank {} finished parallel computation'.format(rank))

        else:
            if self.verbose:
                print("Rank {} computing serially ...".format(rank))
            # List comprehension vs map vs for loop?
            # https://stackoverflow.com/questions/1247486/python-list-comprehension-vs-map
            results = []
            for loops_2d, curr_vdc in self.data:
                results += [self._map_function(vector, curr_vdc) for vector in
                            loops_2d]

        self._results = results

    def compute(self, override=False):
        return super(BELoopProjector, self).compute(self, override=override)

    project_loops = compute

    @staticmethod
    def _map_function(sho_response, dc_offset):
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

    def _write_results_chunk(self):

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

        # Step 1: unzip the two components in results into separate arrays
        loop_mets = np.zeros(shape=len(self._results), dtype=loop_metrics32)
        proj_loops = np.zeros(shape=(len(self._results),
                                     self.dc_offsets_mat.shape[1]),
                              dtype=np.float32)
        ind = 0
        for curr_loop, curr_met in self._results:
            proj_loops[ind] = curr_loop
            loop_mets[ind] = curr_met
            ind += 1

        if self.verbose:
            print('Unzipped results into Projected loops of shape: {} and '
                  'Metrics of shape: {}'.format(proj_loops.shape,
                                                loop_mets.shape))

        # Step 2: Fold to N-D before reversing transposes:


        # Which pixels are we working on?
        curr_pixels = self._get_pixels_in_current_batch()

        self.h5_projected_loops[curr_pixels] = projected_loops_2d2
        self.h5_loop_metrics[curr_pixels] = metrics_2d

        if self.verbose and self.mpi_rank == 0:
            print('Finished ?')