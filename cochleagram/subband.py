from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import numpy as np
from cochleagram import utils


def reshape_signal_canonical(signal):
    if signal.ndim == 1:  # signal is a flattened array
        out_signal = signal
    elif signal.ndim == 2:  # signal is a row or column vector
        if signal.shape[0] == 1:
            out_signal = signal.flatten()
        elif signal.shape[1] == 1:
            out_signal = signal.flatten()
        else:
            raise ValueError('signal must be a row or column vector; found shape: %s' % signal.shape)
    else:
        raise ValueError('signal must be a row or column vector; found shape: %s' % signal.shape)
    return out_signal


def reshape_signal_batch(signal):
    if signal.ndim == 1:  # signal is a flattened array
        out_signal = signal.reshape((1, -1))
    elif signal.ndim == 2:  # signal is a row or column vector
        if signal.shape[0] == 1:
            out_signal = signal
        elif signal.shape[1] == 1:
            out_signal = signal.reshape((1, -1))
        else:  # first dim is batch dim
            out_signal = signal
    else:
        raise ValueError(
            'signal should be flat array, row or column vector, or a 2D matrix with dimensions [batch, waveform]; found %s' % signal.ndim)
    return out_signal


def generate_subband_envelopes_fast(signal, filters, padding_size=None, fft_mode='auto', debug_ret_all=False):
    # convert the signal to a canonical representation
    signal_flat = reshape_signal_canonical(signal)

    if padding_size is not None and padding_size > 1:
        signal_flat, padding = pad_signal(signal_flat, padding_size)

    if np.isrealobj(signal_flat):
        fft_sample = utils.rfft(signal_flat, mode=fft_mode)
        nr = fft_sample.shape[0]
        subbands = np.zeros(filters.shape, dtype=complex)
        subbands[:, :nr] = _real_freq_filter(fft_sample, filters)
    else:
        fft_sample = utils.fft(signal_flat, mode=fft_mode)
        subbands = filters * fft_sample

    analytic_subbands = utils.fhilbert(subbands, mode=fft_mode)
    subband_envelopes = np.abs(analytic_subbands)

    if padding_size is not None and padding_size > 1:
        analytic_subbands = analytic_subbands[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct
        subband_envelopes = subband_envelopes[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct

    if debug_ret_all is True:
        out_dict = {}
        for k in dir():
            if k != 'out_dict':
                out_dict[k] = locals()[k]
        return out_dict
    else:
        return subband_envelopes


def generate_subbands(signal, filters, padding_size=None, fft_mode='auto', debug_ret_all=False):
    signal_flat = reshape_signal_canonical(signal)

    if padding_size is not None and padding_size > 1:
        signal_flat, padding = pad_signal(signal_flat, padding_size)

    is_signal_even = signal_flat.shape[0] % 2 == 0
    if np.isrealobj(signal_flat) and is_signal_even:  # attempt to speed up computation with rfft
        if signal_flat.shape[0] % 2 == 0:
            fft_sample = utils.rfft(signal_flat, mode=fft_mode)
            subbands = _real_freq_filter(fft_sample, filters)
            subbands = utils.irfft(subbands, mode=fft_mode)  # operates row-wise
        else:
            warnings.warn('Consider using even-length signal for a rfft speedup', RuntimeWarning, stacklevel=2)
            fft_sample = utils.fft(signal_flat, mode=fft_mode)
            subbands = filters * fft_sample
            subbands = np.real(utils.ifft(subbands, mode=fft_mode))  # operates row-wise
    else:
        fft_sample = utils.fft(signal_flat, mode=fft_mode)
        subbands = filters * fft_sample
        subbands = np.real(utils.ifft(subbands, mode=fft_mode))  # operates row-wise

    if padding_size is not None and padding_size > 1:
        subbands = subbands[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct

    if debug_ret_all is True:
        out_dict = {}
        for k in dir():
            if k != 'out_dict':
                out_dict[k] = locals()[k]
        return out_dict
    else:
        return subbands


def generate_analytic_subbands(signal, filters, padding_size=None, fft_mode='auto'):
    signal_flat = reshape_signal_canonical(signal)

    if padding_size is not None and padding_size > 1:
        signal_flat, padding = pad_signal(signal_flat, padding_size)

    fft_sample = utils.fft(signal_flat, mode=fft_mode)
    subbands = filters * fft_sample
    analytic_subbands = utils.fhilbert(subbands, mode=fft_mode)

    if padding_size is not None and padding_size > 1:
        analytic_subbands = analytic_subbands[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct

    return analytic_subbands


def generate_subband_envelopes(signal, filters, padding_size=None, debug_ret_all=False):
    analytic_subbands = generate_analytic_subbands(signal, filters, padding_size=padding_size)
    subband_envelopes = np.abs(analytic_subbands)

    if debug_ret_all is True:
        out_dict = {}
        for k in dir():
            if k != 'out_dict':
                out_dict[k] = locals()[k]
        return out_dict
    else:
        return subband_envelopes


def collapse_subbands(subbands, filters, fft_mode='auto'):
    fft_subbands = filters * utils.fft(subbands, mode=fft_mode)
    subbands = np.real(utils.ifft(fft_subbands, mode=fft_mode))
    signal = subbands.sum(axis=0)
    return signal


def pad_signal(signal, padding_size, axis=0):
    if padding_size is not None and padding_size >= 1:
        pad_shape = list(signal.shape)
        pad_shape[axis] = padding_size
        pad_signal = np.concatenate((signal, np.zeros(pad_shape)))
    else:
        padding_size = 0
        pad_signal = signal
    return (pad_signal, padding_size)


def _real_freq_filter(rfft_signal, filters):
    nr = rfft_signal.shape[0]
    subbands = filters[:, :nr] * rfft_signal
    return subbands
