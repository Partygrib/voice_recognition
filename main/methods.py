import math
import time

import cochleagram.cochlea as cgram
import matplotlib.pyplot as plt
import cv2
import librosa
import librosa.display
import librosa.feature
import numpy as np
import matplotlib
from window_slider import Slider

from fractals import katz_fd, petr_fd, cast_fd, hig_fd


def resize_image(file_in, file_out):
    img = cv2.imread(file_in, cv2.IMREAD_GRAYSCALE)
    dim = (128, 128)

    plt.figure(figsize=(12, 3))
    res_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    librosa.display.specshow(res_img)
    plt.gca().invert_yaxis()

    plt.axis('off')
    plt.savefig(file_out, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close('all')

    return res_img


def cochlea(signal1, sr1):
    array = cgram.cochleagram(signal1, sr1, low_lim=40, hi_lim=12000, strict=False, n=256, sample_factor=3)
    matplotlib.use('tkagg')

    Xdb = librosa.amplitude_to_db(abs(array))
    plt.figure(figsize=(12, 3))
    librosa.display.specshow(Xdb, sr=sr1)
    plt.axis('off')
    plt.savefig('pictures/cochleagram.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

    return resize_image('pictures/cochleagram.png', 'pictures/res_cochleagram.png')


def spectrogram(data, sr):
    matplotlib.use('tkagg')

    X = librosa.stft(data, n_fft=512, hop_length=384, win_length=512)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    librosa.display.specshow(Xdb, sr=sr)

    plt.axis('off')
    plt.savefig('pictures/spectrogram.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

    return resize_image('pictures/spectrogram.png', 'pictures/res_spectrogram.png')


def mfcc_f(data, sr):
    matplotlib.use('tkagg')

    S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(abs(S))
    X = librosa.feature.mfcc(S=log_S, n_mfcc=13, dct_type=3)

    plt.figure(figsize=(12, 3))
    librosa.display.specshow(X, sr=sr)
    plt.axis('off')
    plt.savefig('pictures/mfcc.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

    return resize_image('pictures/mfcc.png', 'pictures/res_mfcc.png')


def fractal_matrix(array):
    result1 = []
    result2 = []
    result3 = []
    result4 = []

    splits = np.array_split(array, 512)
    n = 0

    for sp in splits:
        k = katz_fd(sp)
        if math.isnan(k):
            n = n + 1
        p = petr_fd(sp)
        if math.isnan(p):
            n = n + 1
        h = hig_fd(sp)
        if math.isnan(h):
            n = n + 1
        c = cast_fd(sp)
        if math.isnan(c):
            n = n + 1
        result1.append(k)
        result2.append(p)
        result3.append(h)
        result4.append(c)

    if n != 0:
        print('error NaN')

    fd1 = np.array(result1)
    fd2 = np.array(result2)
    fd3 = np.array(result3)
    fd4 = np.array(result4)

    return [fd1, fd2, fd3, fd4]


def create_fd(data):
    array_all = fractal_matrix(data)
    katz_array1 = window_fd(array_all[0], 128, 115)
    katz_array2 = window_fd(array_all[1], 128, 115)
    katz_array3 = window_fd(array_all[2], 128, 115)
    katz_array4 = window_fd(array_all[3], 128, 115)

    katz_array = katz_array1 + katz_array2 + katz_array3 + katz_array4
    fractal_image = np.array(katz_array)

    matplotlib.use('tkagg')
    plt.figure(figsize=(12, 3))
    librosa.display.specshow(fractal_image)
    plt.axis('off')
    plt.savefig('pictures/fractal_image.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

    return fractal_image


def window_fd(array, size, over):
    list = np.array(array)
    bucket_size = size
    overlap_count = over
    slider = Slider(bucket_size, overlap_count)
    slider.fit(list)
    S = []

    first = list[0:128]
    S.append(first)

    while True:
        window_data = slider.slide()
        if slider.reached_end_of_list():
            break
        S.append(window_data)

    last = list[384:512]
    S.append(last)

    return S
