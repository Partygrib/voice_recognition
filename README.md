# voice_recognition
Graduate Qualification Work of a student of SPBPU Khmarenko G.I.
## ABSTRACT
### KEYWORDS: CNN, CBAM, MFCC, SER, NEURAL NETWORK, DEEP LEARNING, SPECTROGRAM, COCHLEAGRAM, FRACTAL DIMENSIONS
The subject of the graduate qualification work is “Development of a system for recognizing emotions by voice".

The work consists of several stages: review of the subject area, familiariza-tion with the theoretical content of the system, technical implementation of the model and summary of the results.

In this paper, we study the sphere of speech recognition of emotions, analyze and identify the main components that are necessary to build a modern SER sys-tem.

In addition, the work provides a detailed study of all the details and features of the model, along with a scheme for the functioning of the system, both individu-ally and as a whole.

Functions and methods were written to extract four features of a speech sig-nal, namely: spectrogram, cochleagram, a set of mel-cepstral coefficients (MFCC) and fractal dimensions, and a 3D CNN architecture with an attention module was implemented.

As a result, 4 models were obtained, trained on 3 datasets (SAVEE, RAV-DESS, TESS) separately and on a mixed sample, which are in many ways not infe-rior in accuracy to current research, and a comparative characteristic is given that proves the importance of using fractal dimensions in the field of deep learning to classify emotions. 

## SYSTEM
![system](https://github.com/Partygrib/voice_recognition/blob/master/resources/model.jpg)

## FEATURES
### Spectrogram

### Cochleagram

### MFCC

### Fractal Dimensions

![features](https://github.com/Partygrib/voice_recognition/blob/master/resources/features.jpg)


## MODEL

![model](https://github.com/Partygrib/voice_recognition/blob/master/resources/architecture.jpg)
