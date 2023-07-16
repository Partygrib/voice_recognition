from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def matlab_arange(start, stop, num):
    return np.linspace(start, stop, num + 1)


def fft(a, n=None, axis=-1, norm=None, mode='auto', params=None):
    mode, params = _parse_fft_mode(mode, params)
    d1 = {'n': n, 'axis': axis, 'norm': norm}
    params = dict(d1, **params)

    if mode == 'fftw':
        import pyfftw
        return pyfftw.interfaces.numpy_fft.fft(a, **params)
    elif mode == 'np':
        return np.fft.fft(a, **params)
    else:
        raise NotImplementedError('`fft method is not defined for mode `%s`;' +
                                  'use "auto", "np" or "fftw".')


def ifft(a, n=None, axis=-1, norm=None, mode='auto', params=None):
    mode, params = _parse_fft_mode(mode, params)
    d1 = {'n': n, 'axis': axis, 'norm': norm}
    params = dict(d1, **params)

    if mode == 'fftw':
        import pyfftw
        return pyfftw.interfaces.numpy_fft.ifft(a, **params)
    elif mode == 'np':
        return np.fft.ifft(a, **params)
    else:
        raise NotImplementedError('`ifft method is not defined for mode `%s`;' +
                                  'use "np" or "fftw".')


def rfft(a, n=None, axis=-1, mode='auto', params=None):
    mode, params = _parse_fft_mode(mode, params)
    d1 = {'n': n, 'axis': axis}
    params = dict(d1, **params)

    if mode == 'fftw':
        import pyfftw
        return pyfftw.interfaces.numpy_fft.rfft(a, **params)
    elif mode == 'np':
        return np.fft.rfft(a, **params)
    else:
        raise NotImplementedError('`rfft method is not defined for mode `%s`;' +
                                  'use "np" or "fftw".')


def irfft(a, n=None, axis=-1, mode='auto', params=None):
    mode, params = _parse_fft_mode(mode, params)
    d1 = {'n': n, 'axis': axis}
    params = dict(d1, **params)

    if mode == 'fftw':
        import pyfftw
        return pyfftw.interfaces.numpy_fft.irfft(a, **params)
    elif mode == 'np':
        return np.fft.irfft(a, **params)
    else:
        raise NotImplementedError('`irfft method is not defined for mode `%s`;' +
                                  'use "np" or "fftw".')


def fhilbert(a, axis=None, mode='auto', ifft_params=None):
    if axis is None:
        axis = np.argmax(a.shape)
    N = a.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    ah = a * h

    return ifft(ah, mode=mode, params=ifft_params)


def _parse_fft_mode(mode, params):
    if mode == 'auto':
        try:
            import pyfftw
            mode = 'fftw'
            if params is None:
                params = {'planner_effort': 'FFTW_ESTIMATE'}  # FFTW_ESTIMATE seems fast
        except ImportError:
            mode = 'np'
            if params is None:
                params = {}
    else:
        if params is None:
            params = {}
    return mode, params