import numpy as np
import sys
from scipy.signal import find_peaks_cwt

sys.path.append(r'C:\Users\Suhas\PycharmProjects\pyUSID')
from pyUSID.io.hdf_utils import get_auxiliary_datasets

from be_sho import SHOestimateGuess, SHOfunc


def reshape_to_one_step(raw_mat, num_steps):
    """
    Reshapes provided data from (pos, step * bin) to (pos * step, bin).
    This is useful when unraveling data for parallel processing.

    Parameters
    -------------
    raw_mat : 2D numpy array
        Data organized as (positions, step * bins)
    num_steps : unsigned int
        Number of spectroscopic steps per pixel (eg - UDVS steps)

    Returns
    --------------
    two_d : 2D numpy array
        Data rearranged as (positions * step, bin)
    """
    num_pos = raw_mat.shape[0]
    num_bins = int(raw_mat.shape[1] / num_steps)
    one_d = raw_mat
    one_d = one_d.reshape((num_bins * num_steps * num_pos))
    two_d = one_d.reshape((num_steps * num_pos, num_bins))
    return two_d


def reshape_to_n_steps(raw_mat, num_steps):
    """
    Reshapes provided data from (positions * step, bin) to (positions, step * bin).
    Use this to restructure data back to its original form after parallel computing

    Parameters
    --------------
    raw_mat : 2D numpy array
        Data organized as (positions * step, bin)
    num_steps : unsigned int
         Number of spectroscopic steps per pixel (eg - UDVS steps)

    Returns
    ---------------
    two_d : 2D numpy array
        Data rearranged as (positions, step * bin)
    """
    num_bins = raw_mat.shape[1]
    num_pos = int(raw_mat.shape[0] / num_steps)
    one_d = raw_mat
    one_d = one_d.reshape(num_bins * num_steps * num_pos)
    two_d = one_d.reshape((num_pos, num_steps * num_bins))
    return two_d


def is_reshapable(h5_main, step_start_inds=None):
    """
    A BE dataset is said to be reshape-able if the number of bins per steps is constant. Even if the dataset contains
    multiple excitation waveforms (harmonics), We know that the measurement is always at the resonance peak, so the
    frequency vector should not change.

    Parameters
    ----------
    h5_main : h5py.Dataset object
        Reference to the main dataset
    step_start_inds : list or 1D array
        Indices that correspond to the start of each BE pulse / UDVS step

    Returns
    ---------
    reshapable : Boolean
        Whether or not the number of bins per step are constant in this dataset
    """
    if step_start_inds is None:
        h5_spec_inds = get_auxiliary_datasets(h5_main, aux_dset_name=['Spectroscopic_Indices'])[0]
        step_start_inds = np.where(h5_spec_inds[0] == 0)[0]
    # Adding the size of the main dataset as the last (virtual) step
    step_start_inds = np.hstack((step_start_inds, h5_main.shape[1]))
    num_bins = np.diff(step_start_inds)
    step_types = np.unique(num_bins)
    return len(step_types) == 1


def r_square(data_vec, func, *args, **kwargs):
    """
    R-square for estimation of the fitting quality
    Typical result is in the range (0,1), where 1 is the best fitting

    Parameters
    ----------
    data_vec : array_like
        Measured data points
    func : callable function
        Should return a numpy.ndarray of the same shape as data_vec
    args :
        Parameters to be pased to func
    kwargs :
        Keyword parameters to be pased to func

    Returns
    -------
    r_squared : float
        The R^2 value for the current data_vec and parameters
    """
    data_mean = np.mean(data_vec)
    ss_tot = sum(abs(data_vec - data_mean) ** 2)
    ss_res = sum(abs(data_vec - func(*args, **kwargs)) ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return r_squared


def wavelet_peaks(vector, peak_width_bounds, peak_width_step=20, **kwargs):
    """
    This is the function that will be mapped by multiprocess. This is a wrapper around the scipy function.
    It uses a parameter - wavelet_widths that is configured outside this function.

    Parameters
    ----------
    vector : 1D numpy array
        Feature vector containing peaks

    Returns
    -------
    peak_indices : list
        List of indices of peaks within the prescribed peak widths
    """
    # The below numpy array is used to configure the returned function wpeaks
    wavelet_widths = np.linspace(peak_width_bounds[0], peak_width_bounds[1], peak_width_step)

    peak_indices = find_peaks_cwt(np.abs(vector), wavelet_widths, **kwargs)

    return peak_indices


def complex_gaussian(resp_vec, w_vec, num_points=5):
        """
        Sets up the needed parameters for the analytic approximation for the
        Gaussian fit of complex data.

        Parameters
        ----------
        resp_vec : numpy.ndarray
            Data vector to be fit.
        args: numpy arrays.

        kwargs: Passed to SHOEstimateFit().

        Returns
        -------
        sho_guess: callable function.

        """
        guess = SHOestimateGuess(resp_vec, w_vec, num_points)
        
        # Calculate the error and append it.
        guess = np.hstack([guess, np.array(r_square(resp_vec, SHOfunc, guess, w_vec))])

        return guess
    

def sho_error(guess, data_vec, freq_vector):
    """
    Generates the single Harmonic Oscillator response over the given vector

    Parameters
    ----------
    guess : array-like
        The set of guess parameters (Amp,w0,Q,phi) to be tested
    data_vec : numpy.ndarray
        The data vector to compare the current guess against
    freq_vector : numpy.ndarray
        The frequencies that correspond to each data point in `data_vec`
    
    Notes
    -----
    Amp: amplitude
    w0: resonant frequency
    Q: Quality Factor
    phi: Phase

    Returns
    -------
    fitness : float
        The 1-r^2 value for the current set of SHO coefficients
    """

    if len(guess) < 4:
        raise ValueError('Error: The Single Harmonic Oscillator requires 4 parameter guesses!')
    
    Amp, w_0, Q, phi = guess[:4]
    guess_vec = Amp * np.exp(1.j * phi) * w_0 ** 2 / (freq_vector ** 2 - 1j * freq_vector * w_0 / Q - w_0 ** 2)

    data_mean = np.mean(data_vec)
    
    ss_tot = np.sum(np.abs(data_vec - data_mean) ** 2)
    ss_res = np.sum(np.abs(data_vec - guess_vec) ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # print('tot: {}\tres: {}\tr2: {}'.format(ss_tot, ss_res, r_squared))

    return 1 - r_squared
