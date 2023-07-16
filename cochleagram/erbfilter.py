from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import numpy as np
from cochleagram import utils


def _identity(x):
    return x


def freq2lin(freq_hz):
    return _identity(freq_hz)


def lin2freq(n_lin):
    return _identity(n_lin)


def freq2erb(freq_hz):
    return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)


def make_cosine_filter(freqs, l, h, convert_to_erb=True):
    if convert_to_erb:
        freqs_erb = freq2erb(freqs)
        l_erb = freq2erb(l)
        h_erb = freq2erb(h)
    else:
        freqs_erb = freqs
        l_erb = l
        h_erb = h

    avg_in_erb = (l_erb + h_erb) / 2  # center of filter
    rnge_in_erb = h_erb - l_erb  # width of filter
    return np.cos((freqs_erb[(freqs_erb > l_erb) & (freqs_erb < h_erb)] - avg_in_erb) / rnge_in_erb * np.pi)


def make_full_filter_set(filts, signal_length=None):
    if signal_length is None:
        signal_length = 2 * filts.shape[1] - 1

    if np.remainder(signal_length, 2) == 0:  # even -- don't take the DC & don't double sample nyquist
        neg_filts = np.flipud(filts[1:filts.shape[0] - 1, :])
    else:  # odd -- don't take the DC
        neg_filts = np.flipud(filts[1:filts.shape[0], :])
    fft_filts = np.vstack((filts, neg_filts))
    return fft_filts.T


def make_ref_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, sample_factor, padding_size=None, full_filter=True,
                            strict=True, ref_spacing_mode='erb', **kwargs):
    ref_spacing_mode = ref_spacing_mode.lower()
    if ref_spacing_mode == 'erb':
        _freq2ref = freq2erb
        _ref2freq = erb2freq
    elif ref_spacing_mode == 'lin' or 'linear':
        _freq2ref = freq2lin
        _ref2freq = lin2freq
    else:
        raise NotImplementedError('unrecognized spacing mode: %s' % ref_spacing_mode)

    if not isinstance(sample_factor, int):
        raise ValueError('sample_factor must be an integer, not %s' % type(sample_factor))
    if sample_factor <= 0:
        raise ValueError('sample_factor must be positive')

    if sample_factor != 1 and np.remainder(sample_factor, 2) != 0:
        msg = 'sample_factor odd, and will change ERB filter widths. Use even sample factors for comparison.'
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if padding_size is not None and padding_size >= 1:
        signal_length += padding_size

    if np.remainder(signal_length, 2) == 0:  # even length
        n_freqs = signal_length // 2  # .0 does not include DC, likely the sampling grid
        max_freq = sr / 2  # go all the way to nyquist
    else:  # odd length
        n_freqs = (signal_length - 1) // 2  # .0
        max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

    if hi_lim > sr / 2:
        hi_lim = max_freq
        msg = 'input arg "hi_lim" exceeds nyquist limit for max frequency; ignore with "strict=False"'
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    n_filters = sample_factor * (n + 1) - 1
    n_lp_hp = 2 * sample_factor
    freqs = utils.matlab_arange(0, max_freq, n_freqs)
    filts = np.zeros((n_freqs + 1, n_filters + n_lp_hp))  # ?? n_freqs+1
    center_freqs, erb_spacing = np.linspace(_freq2ref(low_lim), _freq2ref(hi_lim), n_filters + 2,
                                            retstep=True)  # +2 for bin endpoints
    center_freqs = center_freqs[1:-1]

    freqs_erb = _freq2ref(freqs)
    for i in range(n_filters):
        i_offset = i + sample_factor
        l = center_freqs[i] - sample_factor * erb_spacing
        h = center_freqs[i] + sample_factor * erb_spacing
        filts[(freqs_erb > l) & (freqs_erb < h), i_offset] = make_cosine_filter(freqs_erb, l, h,
                                                                                convert_to_erb=False)

    for i in range(sample_factor):
        i_offset = i + sample_factor
        lp_h_ind = max(
            np.where(freqs < _ref2freq(center_freqs[i]))[0])  # lowpass filter goes up to peak of first cos filter
        lp_filt = np.sqrt(1 - np.power(filts[:lp_h_ind + 1, i_offset], 2))

        hp_l_ind = min(np.where(freqs > _ref2freq(center_freqs[-1 - i]))[
                           0])  # highpass filter goes down to peak of last cos filter
        hp_filt = np.sqrt(1 - np.power(filts[hp_l_ind:, -1 - i_offset], 2))

        filts[:lp_h_ind + 1, i] = lp_filt
        filts[hp_l_ind:, -1 - i] = hp_filt

    filts = filts / np.sqrt(sample_factor)
    cfs_low = np.copy(center_freqs[:sample_factor]) - sample_factor * erb_spacing
    cfs_hi = np.copy(center_freqs[-sample_factor:]) + sample_factor * erb_spacing
    center_freqs = _ref2freq(np.concatenate((cfs_low, center_freqs, cfs_hi)))

    center_freqs[center_freqs < 0] = 1

    if kwargs.get('no_lowpass'):
        filts = filts[:, sample_factor:]
    if kwargs.get('no_highpass'):
        filts = filts[:, :-sample_factor]

    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, center_freqs, freqs


def make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, sample_factor, padding_size=None, full_filter=True,
                            strict=True, **kwargs):
    if not isinstance(sample_factor, int):
        raise ValueError('sample_factor must be an integer, not %s' % type(sample_factor))
    if sample_factor <= 0:
        raise ValueError('sample_factor must be positive')

    if sample_factor != 1 and np.remainder(sample_factor, 2) != 0:
        msg = 'sample_factor odd, and will change ERB filter widths. Use even sample factors for comparison.'
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if padding_size is not None and padding_size >= 1:
        signal_length += padding_size

    if np.remainder(signal_length, 2) == 0:  # even length
        n_freqs = signal_length // 2  # .0 does not include DC, likely the sampling grid
        max_freq = sr / 2  # go all the way to nyquist
    else:  # odd length
        n_freqs = (signal_length - 1) // 2  # .0
        max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

    if hi_lim > sr / 2:
        hi_lim = max_freq
        msg = 'input arg "hi_lim" exceeds nyquist limit for max frequency; ignore with "strict=False"'
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    n_filters = sample_factor * (n + 1) - 1
    n_lp_hp = 2 * sample_factor
    freqs = utils.matlab_arange(0, max_freq, n_freqs)
    filts = np.zeros((n_freqs + 1, n_filters + n_lp_hp))  # ?? n_freqs+1
    center_freqs, erb_spacing = np.linspace(freq2erb(low_lim), freq2erb(hi_lim), n_filters + 2,
                                            retstep=True)  # +2 for bin endpoints
    center_freqs = center_freqs[1:-1]

    freqs_erb = freq2erb(freqs)
    for i in range(n_filters):
        i_offset = i + sample_factor
        l = center_freqs[i] - sample_factor * erb_spacing
        h = center_freqs[i] + sample_factor * erb_spacing
        filts[(freqs_erb > l) & (freqs_erb < h), i_offset] = make_cosine_filter(freqs_erb, l, h, convert_to_erb=False)

    for i in range(sample_factor):
        i_offset = i + sample_factor
        lp_h_ind = max(
            np.where(freqs < erb2freq(center_freqs[i]))[0])  # lowpass filter goes up to peak of first cos filter
        lp_filt = np.sqrt(1 - np.power(filts[:lp_h_ind + 1, i_offset], 2))

        hp_l_ind = min(
            np.where(freqs > erb2freq(center_freqs[-1 - i]))[0])  # highpass filter goes down to peak of last cos filter
        hp_filt = np.sqrt(1 - np.power(filts[hp_l_ind:, -1 - i_offset], 2))

        filts[:lp_h_ind + 1, i] = lp_filt
        filts[hp_l_ind:, -1 - i] = hp_filt

    filts = filts / np.sqrt(sample_factor)
    cfs_low = np.copy(center_freqs[:sample_factor]) - sample_factor * erb_spacing
    cfs_hi = np.copy(center_freqs[-sample_factor:]) + sample_factor * erb_spacing
    center_freqs = erb2freq(np.concatenate((cfs_low, center_freqs, cfs_hi)))

    center_freqs[center_freqs < 0] = 1
    if kwargs.get('no_lowpass'):
        filts = filts[:, sample_factor:]
    if kwargs.get('no_highpass'):
        filts = filts[:, :-sample_factor]

    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, center_freqs, freqs


def make_erb_cos_filters_1x(signal_length, sr, n, low_lim, hi_lim, padding_size=None, full_filter=False, strict=False):
    return make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, 1, padding_size=padding_size,
                                   full_filter=full_filter, strict=strict)


def make_erb_cos_filters_2x(signal_length, sr, n, low_lim, hi_lim, padding_size=None, full_filter=False, strict=False):
    return make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, 2, padding_size=padding_size,
                                   full_filter=full_filter, strict=strict)


def make_erb_cos_filters_4x(signal_length, sr, n, low_lim, hi_lim, padding_size=None, full_filter=False, strict=False):
    return make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, 4, padding_size=padding_size,
                                   full_filter=full_filter, strict=strict)


def make_erb_cos_filters(signal_length, sr, n, low_lim, hi_lim, full_filter=False, strict=False):
    if np.remainder(signal_length, 2) == 0:  # even length
        n_freqs = signal_length / 2  # .0 does not include DC, likely the sampling grid
        max_freq = sr / 2  # go all the way to nyquist
    else:  # odd length
        n_freqs = (signal_length - 1) / 2  # .0
        max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

    freqs = utils.matlab_arange(0, max_freq, n_freqs)
    cos_filts = np.zeros((n_freqs + 1, n))  # ?? n_freqs+1
    a_cos_filts = np.zeros((n_freqs + 1, n))  # ?? n_freqs+1

    if hi_lim > sr / 2:
        hi_lim = max_freq
        if strict:
            raise ValueError('input arg "hi_lim" exceeds nyquist limit for max '
                             'frequency ignore with "strict=False"')

    cutoffs_in_erb = utils.matlab_arange(freq2erb(low_lim), freq2erb(hi_lim), n + 1)  # ?? n+1
    cutoffs = erb2freq(cutoffs_in_erb)

    for k in range(n):
        l = cutoffs[k]
        h = cutoffs[k + 2]  # adjacent filters overlap by 50%
        l_ind = min(np.where(freqs > l)[0])
        h_ind = max(np.where(freqs < h)[0])
        avg = (freq2erb(l) + freq2erb(h)) / 2  # center of filter
        rnge = freq2erb(h) - freq2erb(l)  # width of filter
        cos_filts[l_ind:h_ind + 1, k] = np.cos(
            (freq2erb(freqs[l_ind:h_ind + 1]) - avg) / rnge * np.pi)  # h_ind+1 to include endpoint

    filts = np.zeros((n_freqs + 1, n + 2))
    filts[:, 1:n + 1] = cos_filts
    lp_filt = np.zeros_like(cos_filts[:, :0])
    hp_filt = np.copy(lp_filt)

    filts = np.zeros((n_freqs + 1, n + 2))
    filts[:, 1:n + 1] = cos_filts
    h_ind = max(np.where(freqs < cutoffs[1])[0])  # lowpass filter goes up to peak of first cos filter
    filts[:h_ind + 1, 0] = np.sqrt(1 - filts[:h_ind + 1, 1] ** 2)
    l_ind = min(np.where(freqs > cutoffs[n])[0])  # lowpass filter goes up to peak of first cos filter
    filts[l_ind:n_freqs + 2, n + 1] = np.sqrt(1.0 - filts[l_ind:n_freqs + 2, n] ** 2.0)

    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, cutoffs, freqs


def make_lin_cos_filters(signal_length, sr, n, low_lim, hi_lim, full_filter=False, strict=False):
    if np.remainder(signal_length, 2) == 0:  # even length
        n_freqs = signal_length / 2  # .0 does not include DC, likely the sampling grid
        max_freq = sr / 2  # go all the way to nyquist
    else:  # odd length
        n_freqs = (signal_length - 1) / 2  # .0
        max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

    freqs = utils.matlab_arange(0, max_freq, n_freqs)
    cos_filts = np.zeros((n_freqs + 1, n))  # ?? n_freqs+1
    a_cos_filts = np.zeros((n_freqs + 1, n))  # ?? n_freqs+1

    if hi_lim > sr / 2:
        hi_lim = max_freq
        if strict:
            raise ValueError('input arg "hi_lim" exceeds nyquist limit for max '
                             'frequency ignore with "strict=False"')

    cutoffs = utils.matlab_arange(low_lim, hi_lim, n + 1)  # ?? n+1

    for k in range(n):
        l = cutoffs[k]
        h = cutoffs[k + 2]  # adjacent filters overlap by 50%
        l_ind = min(np.where(freqs > l)[0])
        h_ind = max(np.where(freqs < h)[0])
        avg = (l + h) / 2  # center of filter
        rnge = h - l  # width of filter
        cos_filts[l_ind:h_ind + 1, k] = np.cos(
            (freqs[l_ind:h_ind + 1] - avg) / rnge * np.pi)  # h_ind+1 to include endpoint

    filts = np.zeros((n_freqs + 1, n + 2))
    filts[:, 1:n + 1] = cos_filts
    lp_filt = np.zeros_like(cos_filts[:, :0])
    hp_filt = np.copy(lp_filt)

    filts = np.zeros((n_freqs + 1, n + 2))
    filts[:, 1:n + 1] = cos_filts
    h_ind = max(np.where(freqs < cutoffs[1])[0])  # lowpass filter goes up to peak of first cos filter
    filts[:h_ind + 1, 0] = np.sqrt(1 - filts[:h_ind + 1, 1] ** 2)
    l_ind = min(np.where(freqs > cutoffs[n])[0])  # lowpass filter goes up to peak of first cos filter
    filts[l_ind:n_freqs + 2, n + 1] = np.sqrt(1.0 - filts[l_ind:n_freqs + 2, n] ** 2.0)

    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, cutoffs, freqs
