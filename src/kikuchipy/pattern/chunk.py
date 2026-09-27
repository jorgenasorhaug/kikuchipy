# Copyright 2019-2024 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

"""Private functions for operating on :class:`numpy.ndarray` or
:class:`dask.array.Array` chunks of EBSD patterns.
"""

from typing import Callable

import dask.array as da
from numba import njit
import numpy as np
from scipy.ndimage import correlate

from kikuchipy.filters.window import Window
from kikuchipy.pattern._pattern import _rescale_with_min_max, rescale_intensity
from kikuchipy._constants import verify_dependency_or_raise


def get_dynamic_background(
    patterns: np.ndarray | da.Array,
    filter_func: Callable,
    dtype_out: str | np.dtype | type | None = None,
    **kwargs,
) -> np.ndarray:
    """Obtain the dynamic background in a chunk of EBSD patterns.

    Parameters
    ----------
    patterns
        EBSD patterns.
    filter_func
        Function where a Gaussian convolution filter is applied, in the
        frequency or spatial domain. Either
        :func:`scipy.ndimage.gaussian_filter` or
        :func:`kikuchipy.util.barnes_fftfilter.fft_filter`.
    dtype_out
        Data type of background patterns. If None (default), it is set
        to input patterns' data type.
    **kwargs
        Keyword arguments passed to the Gaussian blurring function
        passed to `filter_func`.

    Returns
    -------
    background : numpy.ndarray
        Large scale variations in the input EBSD patterns.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype
    else:
        dtype_out = np.dtype(dtype_out)

    background = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        background[nav_idx] = filter_func(patterns[nav_idx], **kwargs)

    return background


def fft_filter(
    patterns: np.ndarray,
    filter_func: Callable,
    transfer_function: np.ndarray | Window,
    dtype_out: str | np.dtype | type | None = None,
    **kwargs,
) -> np.ndarray:
    """Filter a chunk of EBSD patterns in the frequency domain.

    Patterns are transformed via the Fast Fourier Transform (FFT) to the
    frequency domain, where their spectrum is multiplied by a filter
    `transfer_function`, and the filtered spectrum is subsequently
    transformed to the spatial domain via the inverse FFT (IFFT).

    Filtered patterns are rescaled to the data type range of
    `dtype_out`.

    Parameters
    ----------
    patterns
        EBSD patterns.
    filter_func
        Function to apply `transfer_function` with.
    transfer_function
        Filter transfer function in the frequency domain.
    dtype_out
        Data type of output patterns. If None (default), it is set to
        the input patterns' data type.
    **kwargs
        Keyword arguments passed to the `filter_func`.

    Returns
    -------
    filtered_patterns
        Filtered EBSD patterns.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype.type
    else:
        dtype_out = np.dtype(dtype_out)

    filtered_patterns = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        filtered_pattern = filter_func(
            patterns[nav_idx], transfer_function=transfer_function, **kwargs
        )

        filtered_patterns[nav_idx] = rescale_intensity(
            filtered_pattern, dtype_out=dtype_out
        )

    return filtered_patterns


def _average_neighbour_patterns(
    patterns: np.ndarray,
    window_sums: np.ndarray,
    window: np.ndarray | Window,
    dtype_out: np.dtype,
    omin: float,
    omax: float,
) -> np.ndarray:
    """See docstring of :func:`average_neighbour_patterns`."""
    patterns = patterns.astype("float32")
    correlated_patterns = correlate(patterns, weights=window, mode="constant")
    rescaled_patterns = _rescale_neighbour_averaged_patterns(
        correlated_patterns, window_sums, dtype_out, omin, omax
    )
    return rescaled_patterns
    

def _estimate_sigma_nlpar(
    patterns: np.ndarray,
    signal_mask: np.ndarray,
) -> np.ndarray:
    """Estimate patterns' approximate standard deviation (sigma) of the 
    noise. 
    
    If :mod:`pyebsdindex` is installed, the 
    :func:`pyebsdindex.nlpar.NLPAR.sigma_numba` is run to calculate the 
    patterns' standard deviation of the noise. If not, the patterns' 
    standard deviation is calculated and returned. 
    
    See docstring of :func:
    `kikuchipy.signals.ebsd.average_non_local_neighbour_patterns`.
    
    Parameters
    ----------
    patterns
        EBSD patterns.
    signal_mask
        A boolean mask of detector pixels to consider in the 
        calculations. (equal to ``False``, i.e. pixels to *mask 
        out* are ``True``.) The mask must be equal to the signal's 
        signal shape. If not given all pixels are used.
        
    Returns
    -------
    sigma
        Patterns' standard deviation.
    """
    if verify_dependency_or_raise("pyebsdindex", "Sigma estimation") is None:
        from pyebsdindex import nlpar
        
        sigma_numba = getattr(nlpar.NLPAR(), "sigma_numba")
        
        pats = np.asarray(patterns, dtype=np.float32)
        
        ndim = np.ndim(pats)
        if ndim == 3:
            nrows, ncols = pats.shape[0], 1
        else:
            nrows, ncols = pats.shape[:2]
        h, w = pats.shape[-2:]
        npix = h * w
        
        if signal_mask is not None:
            if signal_mask.shape != (h,w):
                raise InputError("Invalid signal_mask shape.")
        elif signal_mask is None:
            signal_mask = np.zeros((h,w), dtype = bool)

        indices = np.arange(npix) * ~signal_mask.flatten()
        
        # Line 754: sigma, corr. accumulated squared difference, and 
        # norm. distances for lamdada opt.:
        sigma, _d2, _n2 = sigma_numba(
            pats.reshape((nrows * ncols,) + (-1,)).copy(), 
            nn = 1, # search_radius
            nrows = nrows, 
            ncols = ncols, 
            rowstartcount = np.asarray((0,nrows)), 
            colstartcount = np.asarray((0,ncols)), 
            indices = indices,
            saturation_protect = True
        )
    else:
        # A crude estimate:
        sigma = np.std(patterns, axis=(-2,-1), astype = np.float32)
    
    return sigma.astype(np.float32)
    
def _optimise_lambda(
    patterns: np.ndarray,
    search_radius: int = 1,
    dthresh: float = 0.0,
    signal_mask: np.ndarray | None = None,
    target_weights: tuple | list = (0.5, 0.34, 0.25), 
) -> np.ndarray:
    """This function is essentially the same as 
    :func:`pyebsdindex.nlpar_cpu.opt_lambda_cpu`, but slightly modified 
    to enable optimisation of lambda through 
    :func:`average_non_local_neighbour_patterns`.
    
    Parameters
    ----------
    patterns
        EBSD patterns.
    search_radius
        Pattern nearest neighbour search radius.
    dthresh
        Distance threshold used during NLPAR weighting to 
        suppress neighbours that are too dissimilar.
    signal_mask
        A boolean mask of detector pixels to consider in the 
        calculations. (equal to ``False``, i.e. pixels to *mask 
        out* are ``True``.) The mask must be equal to the signal's 
        signal shape. If not given all pixels are used.'
    target_weights
        Target average weights (over all points in the EBSD scan) 
        that corresponds to an optimised lambda.

    Returns
    -------
    lamopt_values
        Array of optimised lamda corresponding to the target weights.
    """
    # C.f. line 94:
    # will accept all keywords to calcsigma_cpu. 
    # See NLPAR __init__ for default values
    from pyebsdindex import nlpar
    import scipy.optimize as opt
    
    patterns = np.asarray(patterns, dtype=np.float32)

    ny, nx, sy, sx = patterns.shape
    
    target_weights = np.asarray(target_weights)
    dthresh = np.float64(dthresh)

    def loptfunc(lam,d2,tw,dthresh):
      temp = np.maximum(d2, dthresh)
      dw = np.exp(-(temp) / lam ** 2)
      w = np.sum(dw, axis=2) + 1e-12
      metric = np.mean(np.abs(tw - 1.0 / w))
      return metric
        
    nrows = np.uint64(nx)
    ncols = np.uint64(ny) 

    pwidth = np.uint64(sy)
    pheight = np.uint64(sx)
    phw = pheight * pwidth

    nn = search_radius
    nn = np.uint64(nn)

    sigma_numba = getattr(nlpar.NLPAR(), "sigma_numba")

    if signal_mask is None:
        signal_mask = np.ones((sy,sx), bool)
    
    indices = np.arange(
        sy * sx
    ) * signal_mask.flatten()
    
    # C.f. lines 266 & 486:
    # https://github.com/USNavalResearchLaboratory/PyEBSDIndex/blob/main/pyebsdindex/nlpar_cpu.py
    sigma, d2, n2 = sigma_numba(
        patterns.reshape((ny*nx, sy*sx)).copy(), 
        nn = search_radius, 
        nrows = nrows, 
        ncols = ncols, 
        rowstartcount = np.asarray((0,ny)), 
        colstartcount = np.asarray((0,nx)), 
        indices = indices,
    )

    lamopt_values = []
    stride = 1 if sigma.size < 1e6 else 2 #for large scans cut down on the optimization time.
    for tw in target_weights:

        lam = 1.0
        lambopt1 = opt.minimize(
            loptfunc,
            lam,
            args=(d2[0::stride,0::stride,:],tw,dthresh),
            method='Nelder-Mead',
            bounds = [[0.001, 10.0]],
            options={'fatol': 0.0001}
        )

        lamopt_values.append(lambopt1['x'])

    print('', end='')    
    lamopt_values = np.asarray(lamopt_values)

    print("With target weights", target_weights)
    print("The optimised lambda values are", lamopt_values.flatten())
    print("Optimal choice (median): ", np.median(lamopt_values))

    return lamopt_values.flatten()

def _average_non_local_neighbour_patterns(
    patterns: np.ndarray,
    sigma: np.ndarray,
    lamda: int | float,
    window: Union[np.ndarray, Window],
    signal_mask: np.ndarray,
    dtype_out: np.dtype,
    omin: int | float,
    omax: int | float
) -> np.ndarray:
    """See docstring of :func:`average_non_local_neighbour_patterns`."""
    from numpy.lib.stride_tricks import sliding_window_view
    
    pats = patterns.astype("float32")
    nrows, ncols, h, w = pats.shape

    # Center patterns
    center = pats.reshape(
        nrows, ncols, h * w
    )[:, :, ~signal_mask.ravel()]

    # Tot. num. of pixels of interest per pattern
    npix = center.shape[-1]
    """
    if np.mod(window.shape[:2], 2).sum() == 0:
        # The window is not symmetrical wrs. center pattern:
        sr = tuple(
            [
                (window.n_neighbours[0], 
                 window.n_neighbours[1] + 1)
            ] * 2
        )
    elif np.diff(window.shape[:2]) != 0:
        # Asymmetrical window:
        mod = np.mod(window.shape[:2], 2)[::-1]
        sr = (
            (window.shape[0] // 2 - mod[0], window.shape[0] // 2),
            (window.shape[1] // 2 - mod[1], window.shape[1] // 2)
        )
    else:
        # Symmetrical window
        sr = tuple(
            [
                (window.n_neighbours[0], 
                 window.n_neighbours[1])
            ] * 2
        )"""
        
    win = window.shape[:2]
    
    sr = tuple(
        (
            size // 2,
            size - 1 - size // 2,
        )
        for size in win
    )
    
    # Preparing patterns, neighbour patterns, and sigmas
    padded_pat = np.pad(
        array = center,
        pad_width = (sr[0], sr[1], (0, 0)),
        mode="edge",
    )
    
    padded_sigma = np.pad(
        array = sigma,
        pad_width = (sr[0], sr[1], (0, 0), (0, 0)),
        mode="edge",
    )

    neigh = sliding_window_view(
        padded_pat,
        win,
        axis=(0, 1),
    )
    
    neigh = np.moveaxis(
        neigh,
        (-2,-1),
        (2,3)
    ).reshape(
        nrows,
        ncols,
        np.prod(win),
        npix,
    )#[:,:,window.squeeze().astype(bool),:]
    
    win_mask = np.asarray(
        window[:,:,0,0],
        dtype = bool
    ).ravel()
    
    neigh = neigh[:,:, win_mask,:]
    
    neigh_sigma = sliding_window_view(
        padded_sigma,
        win,
        axis=(0, 1),
    )
    
    # (X, Y, 1, 1, x1+x2, y1+y2)
    neigh_sigma = neigh_sigma.reshape(
        nrows,
        ncols,
        np.prod(win),
    )[:,:, win_mask]#[..., window.astype(bool).flatten()]
    
    # Expanding the dimensions
    center = center[:, :, None, :]
    
    # Euclidean distance | Eq. 3
    diff = neigh - center

    d2 = np.sum(diff * diff, axis=-1, dtype = np.float32)
    
    # Sum of sigmas squared | c.f. Eq. 7
    sig2 = (
        sigma[:, :, 0, 0, None] ** 2
        + neigh_sigma ** 2
    )
    
    # Normalised distance | Eq. 7
    dnorm = (
        d2 - (npix * sig2)
    ) / (
        sig2 * np.sqrt(2.0 * npix)
    )
    
    # Weights | Eq. 8
    weights = np.exp(
        -np.maximum(dnorm, 0.0)
        / (lamda * lamda)
    ).astype("float32")
    
    weights /= (
        weights.sum(axis=-1, keepdims=True)
        + 1e-12 # 1e-6
    )
    
    # Processed pattern vectors | Eq. 1
    pats[:, :, ~signal_mask] = np.sum(
        neigh * weights[..., None],
        axis=2,
    )
    
    rescaled_patterns = _rescale_neighbour_averaged_patterns(
        patterns = pats, 
        window_sums = np.ones((nrows, ncols)),#weights.sum(axis = -1), 
        dtype_out = dtype_out, 
        omin = omin, 
        omax = omax
    )

    return rescaled_patterns


@njit(cache=True, fastmath=True, nogil=True)
def _rescale_neighbour_averaged_patterns(
    patterns: np.ndarray,
    window_sums: np.ndarray,
    dtype_out: np.dtype,
    omin: float,
    omax: float,
) -> np.ndarray:
    """See docstring of :func:`average_neighbour_patterns`."""
    rescaled_patterns = np.zeros(patterns.shape, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        pattern_i = patterns[nav_idx] / window_sums[nav_idx]
        imin = np.min(pattern_i)
        imax = np.max(pattern_i)
        rescaled_patterns[nav_idx] = _rescale_with_min_max(
            pattern_i, imin, imax, omin, omax
        )
    return rescaled_patterns
