from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.signal
import cochleagram.erbfilter as erb
import cochleagram.subband as sb


def cochleagram(signal, sr, n, low_lim, hi_lim, sample_factor,
                padding_size=None, downsample=None, nonlinearity=None,
                fft_mode='auto', ret_mode='envs', strict=True, **kwargs):
    if strict:
        if not isinstance(sr, int):
            raise ValueError('`sr` must be an int; ignore with `strict`=False')
        # make sure low_lim and hi_lim are int
        if not isinstance(low_lim, int):
            raise ValueError('`low_lim` must be an int; ignore with `strict`=False')
        if not isinstance(hi_lim, int):
            raise ValueError('`hi_lim` must be an int; ignore with `strict`=False')

    ret_mode = ret_mode.lower()
    if ret_mode == 'all':
        ret_all_sb = True
    else:
        ret_all_sb = False

    # verify n is positive
    if n <= 0:
        raise ValueError('number of filters `n` must be positive; found: %s' % n)

    # allow for batch generation without creating filters everytime
    batch_signal = sb.reshape_signal_batch(signal)  # (batch_dim, waveform_samples)

    # only make the filters once
    if kwargs.get('no_hp_lp_filts'):
        erb_kwargs = {'no_highpass': True, 'no_lowpass': True}
    else:
        erb_kwargs = {}
    filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(batch_signal.shape[1],
                                                           sr, n, low_lim, hi_lim, sample_factor,
                                                           padding_size=padding_size,
                                                           full_filter=True, strict=strict, **erb_kwargs)
    freqs_to_plot = np.log10(freqs)

    is_batch = batch_signal.shape[0] > 1
    for i in range(batch_signal.shape[0]):
        temp_signal_flat = sb.reshape_signal_canonical(batch_signal[i, ...])

        if ret_mode == 'envs' or ret_mode == 'all':
            temp_sb = sb.generate_subband_envelopes_fast(temp_signal_flat, filts,
                                                         padding_size=padding_size, fft_mode=fft_mode,
                                                         debug_ret_all=ret_all_sb)
        elif ret_mode == 'subband':
            temp_sb = sb.generate_subbands(temp_signal_flat, filts, padding_size=padding_size,
                                           fft_mode=fft_mode, debug_ret_all=ret_all_sb)
        elif ret_mode == 'analytic':
            temp_sb = sb.generate_subbands(temp_signal_flat, filts, padding_size=padding_size,
                                           fft_mode=fft_mode)
        else:
            raise NotImplementedError('`ret_mode` is not supported.')

        if ret_mode == 'envs':
            if downsample is None or callable(downsample):
                # downsample is None or callable
                temp_sb = apply_envelope_downsample(temp_sb, downsample)
            else:
                # interpret downsample as new sampling rate
                temp_sb = apply_envelope_downsample(temp_sb, 'poly', sr, downsample)
            temp_sb = apply_envelope_nonlinearity(temp_sb, nonlinearity)

        if i == 0:
            sb_out = np.zeros(([batch_signal.shape[0]] + list(temp_sb.shape)))
        sb_out[i] = temp_sb

    sb_out = sb_out.squeeze()
    if ret_mode == 'all':
        out_dict = {}
        # add all local variables to out_dict
        for k in dir():
            if k != 'out_dict':
                out_dict[k] = locals()[k]
        return out_dict
    else:
        return sb_out


def apply_envelope_downsample(subband_envelopes, mode, audio_sr=None, env_sr=None, invert=False, strict=True):
    if mode is None:
        pass
    elif callable(mode):
        # apply the downsampling function
        subband_envelopes = mode(subband_envelopes)
    else:
        mode = mode.lower()
        if audio_sr is None:
            raise ValueError('`audio_sr` cannot be None. Provide sampling rate of original audio signal.')
        if env_sr is None:
            raise ValueError('`env_sr` cannot be None. Provide sampling rate of subband envelopes (cochleagram).')

        if mode == 'decimate':
            if invert:
                raise NotImplementedError()
            else:
                # was BadCoefficients error with Chebyshev type I filter [default]
                subband_envelopes = scipy.signal.decimate(subband_envelopes, audio_sr // env_sr, axis=1,
                                                          ftype='fir')  # this caused weird banding artifacts
        elif mode == 'resample':
            if invert:
                subband_envelopes = scipy.signal.resample(subband_envelopes,
                                                          np.ceil(subband_envelopes.shape[1] * (audio_sr / env_sr)),
                                                          axis=1)  # fourier method: this causes NANs that get converted to 0s
            else:
                subband_envelopes = scipy.signal.resample(subband_envelopes,
                                                          np.ceil(subband_envelopes.shape[1] * (env_sr / audio_sr)),
                                                          axis=1)  # fourier method: this causes NANs that get converted to 0s
        elif mode == 'poly':
            if strict:
                n_samples = subband_envelopes.shape[1] * (audio_sr / env_sr) if invert else subband_envelopes.shape[
                                                                                                1] * (env_sr / audio_sr)
                if not np.isclose(n_samples, int(n_samples)):
                    raise ValueError(
                        'Choose `env_sr` and `audio_sr` such that the number of samples after polyphase resampling is an integer' +
                        '\n(length: %s, env_sr: %s, audio_sr: %s !--> %s' % (
                        subband_envelopes.shape[1], env_sr, audio_sr, n_samples))
            if invert:
                subband_envelopes = scipy.signal.resample_poly(subband_envelopes, audio_sr, env_sr,
                                                               axis=1)  # this requires v0.18 of scipy
            else:
                subband_envelopes = scipy.signal.resample_poly(subband_envelopes, env_sr, audio_sr,
                                                               axis=1)  # this requires v0.18 of scipy
        else:
            raise ValueError('Unsupported downsampling `mode`: %s' % mode)
    subband_envelopes[subband_envelopes < 0] = 0
    return subband_envelopes


def apply_envelope_nonlinearity(subband_envelopes, nonlinearity, invert=False):
    # apply nonlinearity
    if nonlinearity is None:
        pass
    elif nonlinearity == "power":
        if invert:
            subband_envelopes = np.power(subband_envelopes, 10.0 / 3.0)  # from Alex's code
        else:
            subband_envelopes = np.power(subband_envelopes, 3.0 / 10.0)  # from Alex's code
    elif nonlinearity == "db":
        if invert:
            subband_envelopes = np.power(10, subband_envelopes / 20)  # adapted from Anastasiya's code
        else:
            dtype_eps = np.finfo(subband_envelopes.dtype).eps
            subband_envelopes[subband_envelopes == 0] = dtype_eps
            subband_envelopes = 20 * np.log10(subband_envelopes / np.max(subband_envelopes))
            subband_envelopes[subband_envelopes < -60] = -60
    elif callable(nonlinearity):
        subband_envelopes = nonlinearity(subband_envelopes)
    else:
        raise ValueError('argument "nonlinearity" must be "power", "db", or a function.')
    return subband_envelopes
