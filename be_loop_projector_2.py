import numpy as np
import joblib

from pyUSID.io.hdf_utils import copy_region_refs, write_simple_attrs, \
    create_results_group, write_reduced_anc_dsets, \
    create_empty_dataset, write_main_dataset, get_attr, get_unit_values, \
    reshape_to_n_dims, get_sort_order
from pyUSID.io.usi_data import USIDataset
from pyUSID.processing.process import Process
from pyUSID.processing.comp_utils import get_MPI

# From this project:
from be_sho_fitter import sho32
from be_loop import projectLoop

'''
Custom dtype for the datasets created during fitting.
'''

loop_metrics32 = np.dtype({'names': ['Area', 'Centroid x', 'Centroid y',
                                     'Rotation Angle [rad]', 'Offset'],
                           'formats': [np.float32, np.float32, np.float32,
                                       np.float32, np.float32]})


class BELoopProjector(Process):

    def __init__(self, h5_main, **kwargs):
        super(BELoopProjector, self).__init__(h5_main, **kwargs)

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

        self.process_name = "Loop_Projection"
        self.parms_dict = {'projection_method': 'pycroscopy BE loop model'}

    def _create_results_datasets(self):
        """
        Setup the Loop_Fit Group and the loop projection datasets

        """
        # First grab the spectroscopic indices and values and position indices
        # TODO: Avoid unnecessary namespace pollution
        # self._sho_spec_inds = self.h5_main.h5_spec_inds
        # self._sho_spec_vals = self.h5_main.h5_spec_vals
        # self._sho_pos_inds = self.h5_main.h5_pos_inds

        # Which row in the spec datasets is DC offset?
        self._fit_spec_index = self.h5_main.spec_dim_labels.index(
            self._fit_dim_name)

        # TODO: Unkown usage of variable. Waste either way
        # self._fit_offset_index = 1 + self._fit_spec_index

        # Calculate the number of loops per position
        cycle_start_inds = np.argwhere(
            self.h5_main.h5_spec_inds[self._fit_spec_index, :] == 0).flatten()
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
            basename='Loop_Metrics', verbose=self.verbose)

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

        spec_dim_order_s2f = get_sort_order(self.h5_main.h5_spec_inds)[::-1]

        # order_to_s2f = list(pos_dim_order_s2f) + list( len(pos_dim_order_s2f) + spec_dim_order_s2f)
        order_to_s2f = [0] + list(1 + spec_dim_order_s2f)
        print('Order for reshaping to S2F: {}'.format(order_to_s2f))

        self._dim_labels_s2f = list(['Positions']) + list(
            np.array(self.h5_main.spec_dim_labels)[spec_dim_order_s2f])

        print(self._dim_labels_s2f, order_to_s2f)

        self._num_forcs = int(any([targ in self.h5_main.spec_dim_labels for targ in
                             ['FORC', 'FORC_Cycle']]))
        if self._num_forcs:
            forc_pos = self.h5_main.spec_dim_labels.index(self._forc_dim_name)
            self._num_forcs = self.h5_main.spec_dim_sizes[forc_pos]
        print('Num FORCS: {}'.format(self._num_forcs))

        all_but_forc_rows = []
        for ind, dim_name in enumerate(self.h5_main.spec_dim_labels):
            if dim_name not in ['FORC', 'FORC_Cycle', 'FORC_repeat']:
                all_but_forc_rows.append(ind)
        print('All but FORC rows: {}'.format(all_but_forc_rows))

        dc_mats = []

        forc_mats = []

        num_reps = 1 if self._num_forcs == 0 else self._num_forcs
        for forc_ind in range(num_reps):
            print('')
            print('Working on FORC #{}'.format(forc_ind))
            if self._num_forcs:
                this_forc_spec_inds = \
                np.where(self.h5_main.h5_spec_inds[forc_pos] == forc_ind)[0]
            else:
                this_forc_spec_inds = np.ones(
                    shape=self.h5_main.h5_spec_inds.shape[1], dtype=np.bool)

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

            this_forc_2d = self.h5_main[:, this_forc_spec_inds]
            print(
                '2D slice shape for this FORC: {}'.format(this_forc_2d.shape))
            """
            this_forc_nd, success = reshape_to_n_dims(this_forc_2d, 
                                                      h5_pos=self.h5_main.h5_pos_inds[:,:], # THis line will need to change
                                                      h5_spec=self.h5_main.h5_spec_inds[:, this_forc_spec_inds])
            """
            this_forc_nd, success = reshape_to_n_dims(this_forc_2d,
                                                      h5_pos=None,
                                                      # THis line will need to change
                                                      h5_spec=self.h5_main.h5_spec_inds[
                                                              :,
                                                              this_forc_spec_inds])
            print(this_forc_nd.shape)

            this_forc_nd_s2f = this_forc_nd.transpose(
                order_to_s2f).squeeze()  # squeeze out FORC
            dim_names_s2f = self._dim_labels_s2f.copy()
            if self._num_forcs > 0:
                dim_names_s2f.remove(
                    self._forc_dim_name)  # because it was never there in the first place.
            print('Reordered to S2F: {}, {}'.format(this_forc_nd_s2f.shape,
                                                    dim_names_s2f))

            rest_dc_order = list(range(len(dim_names_s2f)))
            _dc_ind = dim_names_s2f.index(self._fit_dim_name)
            rest_dc_order.remove(_dc_ind)
            rest_dc_order = rest_dc_order + [_dc_ind]
            print('Transpose for reordering to rest, DC: {}'.format(
                rest_dc_order))

            rest_dc_nd = this_forc_nd_s2f.transpose(rest_dc_order)
            rest_dc_names = list(np.array(dim_names_s2f)[rest_dc_order])

            self._pre_flattening_shape = list(rest_dc_nd.shape)
            self._pre_flattening_dim_name_order = list(rest_dc_names)

            print('After reodering: {}, {}'.format(rest_dc_nd.shape,
                                                   rest_dc_names))

            dc_rest_2d = rest_dc_nd.reshape(np.prod(rest_dc_nd.shape[:-1]),
                                            np.prod(rest_dc_nd.shape[-1]))
            print('Shape after flattening to 2D: {}'.format(dc_rest_2d.shape))
            forc_mats.append(dc_rest_2d)

            self.data = forc_mats, dc_mats

    def _unit_computation(self):
        if self.verbose and self.mpi_rank == 0:
            print("Rank {} at custom _unit_computation".format(self.mpi_rank))

        resp_2d_list, dc_vec_list = self.data

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
            for loops_2d, curr_vdc in zip(resp_2d_list, dc_vec_list):
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
            for loops_2d, curr_vdc in zip(resp_2d_list, dc_vec_list):
                results += [self._map_function(vector, curr_vdc) for vector in
                            loops_2d]

        self._results = results

    def compute(self, override=False):
        return super(BELoopProjector, self).compute(override=override)

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
                                     self.data[0][0].shape[1]),
                              dtype=np.float32)

        if self.verbose:
            print('Prepared empty arrays for loop metrics of shape: {} and '
                  'projected loops of shape: {}.'
                  ''.format(loop_mets.shape, proj_loops.shape))

        for ind in range(len(self._results)):
            proj_loops[ind] = self._results[ind][0]
            loop_mets[ind] = self._results[ind][1]

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

        # Which pixels are we working on?
        curr_pixels = self._get_pixels_in_current_batch()

        if self.verbose:
            print('Writing projected loops of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(loops_2d.shape, loops_2d.dtype, self.h5_projected_loops.shape, self.h5_projected_loops.dtype))
            print('Writing loop metrics of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(mets_2d.shape, mets_2d.dtype, self.h5_loop_metrics.shape, self.h5_loop_metrics.dtype))

        self.h5_projected_loops[curr_pixels, :] = loops_2d
        self.h5_loop_metrics[curr_pixels, :] = mets_2d

        """
        if self.verbose and self.mpi_rank == 0:
            print('Finished ?')
        """
