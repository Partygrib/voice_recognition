# voice_recognition
Graduate Qualification Work of a student of SPBPU Khmarenko G.I.
## Abstract
### KEYWORDS: CNN, CBAM, MFCC, SER, NEURAL NETWORK, DEEP LEARNING, SPECTROGRAM, COCHLEAGRAM, FRACTAL DIMENSIONS
The subject of the graduate qualification work is “Development of a system for recognizing emotions by voice".

The work consists of several stages: review of the subject area, familiariza-tion with the theoretical content of the system, technical implementation of the model and summary of the results.

In this paper, we study the sphere of speech recognition of emotions, analyze and identify the main components that are necessary to build a modern SER sys-tem.

In addition, the work provides a detailed study of all the details and features of the model, along with a scheme for the functioning of the system, both individu-ally and as a whole.

Functions and methods were written to extract four features of a speech sig-nal, namely: spectrogram, cochleagram, a set of mel-cepstral coefficients (MFCC) and fractal dimensions, and a 3D CNN architecture with an attention module was implemented.

As a result, 4 models were obtained, trained on 3 datasets (SAVEE, RAV-DESS, TESS) separately and on a mixed sample, which are in many ways not infe-rior in accuracy to current research, and a comparative characteristic is given that proves the importance of using fractal dimensions in the field of deep learning to classify emotions.

## How the system work
![system](https://github.com/Partygrib/voice_recognition/blob/master/resources/model.jpg)

The input is an audio recording with a human voice. Further, at the preprocessing stage, the recording is converted into a digital signal to extract the following characteristics: spectrogram, cochleagram, chalk-cepstral coefficients (MFCC) and fractal images. Each of these features is an image that is scaled to 128 x 128 pixels. This is provided to reduce the computational complexity of the entire preprocessing step. After that, an input array of 128 x 128 x 4 x 1 is formed. This array, which is the equivalent of an audio signal, is input to a pre-trained complex 3D CNN model with an attention mechanism for mood classification.

## Features
- Spectrogram
- Cochleagram
- MFCC
- Fractal Dimensions:
  + Katz
  + Higuchi
  + Petrosian
  + Castigloni

As an image of the spectrogram and MFCC, implementations from the librosa library are used. Because these are the most popular methods for representing an audio signal. The cochleagram and methods for calculating fractal dimensions are described locally within the framework of this work.

To construct the spectrogram image, a Hanning window with a length of 512 samples was chosen, the overlap was 384, and the number of FFTs was 512, respectively. To create the MFCC, 13 triangular filters were used to produce a 13th degree Chalk cepstrum. To construct the cochleagram, 256 auditory filters were used, covering the frequency range from 40 to 12 kHz.

The fractal dimensions are calculated from the input speech signals in fractals.py, which contains the implementation of all 4 methods, and then the calculated fractal dimensions are combined in the rows of the matrix to form images in the fractal_matrix function. All four approaches to calculating fractal dimensions use the 512 sliding window method described by the window_fd function. The k_max value for the Higuchi fractal dimension was chosen to be 5.

![features](https://github.com/Partygrib/voice_recognition/blob/master/resources/features.jpg)

resize_function scales all rendered images to 128 x 128 pixels. This conversion is provided to reduce the computational complexity of the system, and also allows you to bring all images under a single format.


## Model architecture
The core of the whole system is a trained model of a 3D convolutional neural network (CNN) with attention module (CBAM)

![model](https://github.com/Partygrib/voice_recognition/blob/master/resources/architecture.jpg)
