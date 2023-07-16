import os
import time

import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from attention import cbam_block
from methods import cochlea, spectrogram, mfcc_f, create_fd


def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Signal'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def feature_extraction(path):
    start = time.perf_counter()
    data, sampling_rate = librosa.load(path, duration=2.5, offset=0.6)
    spec_array = np.array(spectrogram(data, sampling_rate), dtype=float)
    cochlea_array = np.array(cochlea(data, sampling_rate), dtype=float)
    mfcc_array = np.array(mfcc_f(data, sampling_rate), dtype=float)
    fractal_array = np.array(create_fd(data), dtype=float)

    result_array = np.array(np.dstack((spec_array, cochlea_array, mfcc_array, fractal_array)))
    result = np.array([result_array]).reshape(128, 128, 4, 1)
    end = time.perf_counter()
    print(end - start)

    return result


def tess_ds(path):
    tess_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(path + dir)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part == 'ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(path + dir + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)

    return Tess_df


def ravdess_ds(path):
    ravdess_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        actor = os.listdir(path + dir)
        for file in actor:
            part = file.split('.')[0]
            part1 = part.split('-')
            file_emotion.append(int(part1[2]))
            file_path.append(path + dir + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    Ravdess_df.Emotions.replace(
        {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
        inplace=True)

    return Ravdess_df


def savee_ds(path):
    savee_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(path + file)
        part = file.split('_')[1]
        ele = part[:-6]
        if ele == 'a':
            # file_emotion.append('angry')
            file_emotion.append(0)
        elif ele == 'd':
            # file_emotion.append('disgust')
            file_emotion.append(1)
        elif ele == 'f':
            # file_emotion.append('fear')
            file_emotion.append(2)
        elif ele == 'h':
            # file_emotion.append('happy')
            file_emotion.append(3)
        elif ele == 'n':
            # file_emotion.append('neutral')
            file_emotion.append(4)
        elif ele == 'sa':
            # file_emotion.append('sad')
            file_emotion.append(5)
        else:
            # file_emotion.append('surprise')
            file_emotion.append(6)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Savee_df = pd.concat([emotion_df, path_df], axis=1)

    return Savee_df


def y_to_int(Y):
    Y_int = []

    for ele in Y:
        if ele == 'angry':
            Y_int.append(0)
        elif ele == 'disgust':
            Y_int.append(1)
        elif ele == 'fear':
            Y_int.append(2)
        elif ele == 'happy':
            Y_int.append(3)
        elif ele == 'neutral':
            Y_int.append(4)
        elif ele == 'sad':
            Y_int.append(5)
        elif ele == 'calm':
            Y_int.append(7)
        else:
            # surprise
            Y_int.append(6)

    return Y_int


def ds_preprocess(x_name, y_name, path):
    ds = savee_ds(path)
    X = []
    Y = []
    k = 0

    for emotion, path in zip(ds.Emotions, ds.Path):
        feature = feature_extraction(path)
        X.append(feature)
        Y.append(emotion)

        k = k + 1
        print(k)

    x_data = np.array(X)
    y_data = np.array(Y)
    print(x_data.shape)

    np.save(x_name, x_data)
    np.save(y_name, y_data)


# loading preprocessing datasets
x_train_tes = np.load('ds/complete_ds/TESS/train_x.npy')
y_train_tes = np.load('ds/complete_ds/TESS/train_y.npy')
x_test_tes = np.load('ds/complete_ds/TESS/test_x.npy')
y_test_tes = np.load('ds/complete_ds/TESS/test_y.npy')

x_train_rav = np.load('ds/complete_ds/RAVDESS/train_x.npy')
y_train_rav = np.load('ds/complete_ds/RAVDESS/train_y.npy')
x_test_rav = np.load('ds/complete_ds/RAVDESS/test_x.npy')
y_test_rav = np.load('ds/complete_ds/RAVDESS/test_y.npy')

x_train_sav = np.load('ds/complete_ds/SAVEE/train_x.npy')
y_train_sav = np.load('ds/complete_ds/SAVEE/train_y.npy')
x_test_sav = np.load('ds/complete_ds/SAVEE/test_x.npy')
y_test_sav = np.load('ds/complete_ds/SAVEE/test_y.npy')

#x_train = np.concatenate([x_train_tes, x_train_rav, x_train_sav], axis=0)
x_train = x_train_rav
x_test = x_test_rav
#x_test = np.concatenate([x_test_tes, x_test_rav, x_test_sav], axis=0)
y_train = y_train_rav
#y_train = np.concatenate([y_train_tes, y_train_rav, y_train_sav], axis=0)
y_test = y_test_rav
#y_test = np.concatenate([y_test_tes, y_test_rav, y_test_sav], axis=0)

y_train = to_categorical(y_train).astype(np.integer)
y_test = to_categorical(y_test).astype(np.integer)

sample_shape = (128, 128, 4, 1)


def model3DCNN():
    inputs = Input(sample_shape)

    # 1
    conv1 = Conv3D(32, kernel_size=(3, 3, 1), kernel_initializer='he_uniform')(inputs)
    bn1 = BatchNormalization(center=True, scale=True)(conv1)
    act1 = Activation("relu")(bn1)
    maxpool1 = MaxPooling3D(pool_size=(2, 2, 1))(act1)
    # 2
    conv2 = Conv3D(64, kernel_size=(3, 3, 1), kernel_initializer='he_uniform')(maxpool1)
    bn2 = BatchNormalization(center=True, scale=True)(conv2)
    act2 = Activation("relu")(bn2)
    maxpool2 = MaxPooling3D(pool_size=(3, 3, 1))(act2)
    # 3
    conv3 = Conv3D(128, kernel_size=(3, 3, 1), kernel_initializer='he_uniform')(maxpool2)
    act3 = Activation("relu")(conv3)
    # 4
    conv4 = Conv3D(128, kernel_size=(3, 3, 1), kernel_initializer='he_uniform')(act3)  # 65
    maxpool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)
    # 5
    conv5 = Conv3D(256, kernel_size=(3, 3, 1), kernel_initializer='he_uniform')(maxpool4)
    act5 = Activation("relu")(conv5)
    # 6
    conv6 = Conv3D(512, kernel_size=(3, 3, 1), kernel_initializer='he_uniform')(act5)
    attn = cbam_block(conv6)

    flattened = Flatten()(attn)

    fc1 = Dense(units=512, activation='relu')(flattened)
    drop3 = Dropout(rate=0.5)(fc1)

    final = Dense(units=8, activation='softmax')(drop3)

    model = Model(inputs=inputs, outputs=final)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
                        batch_size=12,
                        epochs=30,
                        validation_data=(x_test, y_test),
                        verbose=1)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # model.save('pretrained_models/complete_model_30')


def prediction_emotions(signal):
    pretrained_model = keras.models.load_model('pretrained_models/complete_model_30')

    p_array = np.array([feature_extraction(signal)])
    prediction = np.array(pretrained_model.predict(p_array)).reshape(8, 1)

    print("Result:")
    print('angry: %.2f' % (prediction[0] * 100), '%')
    print('disgust: %.2f' % (prediction[1] * 100), '%')
    print('fear: %.2f' % (prediction[2] * 100), '%')
    print('happy: %.2f' % (prediction[3] * 100), '%')
    print('neutral: %.2f' % (prediction[4] * 100), '%')
    print('sad: %.2f' % (prediction[5] * 100), '%')
    print('surprise: %.2f' % (prediction[6] * 100), '%')
    print('calm: %.2f' % (prediction[7] * 100), '%')


if __name__ == '__main__':
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    feature_extraction('DC_a01.wav')
    #model3DCNN()
    #prediction_emotions(some_path)
