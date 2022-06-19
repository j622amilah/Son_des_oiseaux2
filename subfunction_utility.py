# Subfunctions : Created by Jamilah Foucher June 18, 2021
import os
from os import path

import numpy as np
from numpy.linalg import norm

import pandas as pd

# Signal processing
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy import signal
from scipy.signal import periodogram
from scipy.signal import find_peaks

# Audio processing
from pydub import AudioSegment

# MFCC
import librosa, librosa.display

# Wavelet 
import pywt
from pywt import wavedec

# Visualization
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import IPython

# to display full dataframe information
pd.set_option('display.max_colwidth', None)

from PIL import Image

from mlxtend.preprocessing import minmax_scaling

import pickle

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
tf.compat.v1.enable_eager_execution()  # This allows you to use placeholder in version 2.0 or higher
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

# LSTM
from tensorflow.keras.layers import LSTM, Dense

# Transformer
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization

# N-Layer NN
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization

# CNN
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, GlobalMaxPool2D, MaxPool2D, concatenate, Activation)

# -----------------------------------

def catstrvec_2_catnumvec(strcats, df_col):
    act_val = []
    for i in df_col.to_numpy():
        xy, x_ind, y_ind = np.intersect1d(strcats, i, return_indices=True)
        act_val.append(x_ind[0])
    
    return act_val

# -----------------------------------

def convert_mp3_to_sig(audio_filepath):
    # files                                                                         
    src = audio_filepath
    dst = "test.wav"

    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
    
    return dst

# -----------------------------------

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# ----------------------------------- 
    
# Calculate and plot spectrogram for a wav audio file
# Binning the time-series and calculating the periodogram per time bin

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    # print('size of extracted data from sound file: ', data.shape)
    nfft = 100 # Length of each window segment - frequency data will be binned by nfft/2 
    fs = 8000 # Sampling frequencies
    noverlap = 0 #120 # Overlap between windows
    nchannels = data.ndim
    # print('nchannels: ', nchannels)
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    
    # OR
    
    # pxx = periodogram per frequency (nfft/2 = num_of_freqs, bins=num_of_bins)
    # freqs = freqencies that correspond to each magnitude
    # bins = number of bins to group time-series
    # im = the image of the axis
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap=noverlap)
        
    # print('number of bins: ', bins.shape)
    # print('shape of pxx: ', pxx.shape)
    # print('shape of freqs: ', freqs.shape)
    
    plotORnot = 0
    if plotORnot == 1:
        plt.ylabel("Frequency")
        plt.xlabel("Number of time windows")
        plt.title("Spectrogram")
        plt.show()

    return data, pxx, freqs
       
# -----------------------------------

def make_a_properlist(vec):
    
    out = []
    for i in range(len(vec)):
        out = out + [np.ravel(vec[i])]
        
    if is_empty(out) == False:
        vecout = np.concatenate(out).ravel().tolist()
    else:
        vecout = list(np.ravel(out))
    
    return vecout

def is_empty(vec):
    vec = np.array(vec)
    if vec.shape[0] == 0:
        out = True
    else:
        out = False
        
    return out

# -----------------------------------

def linear_intercurrentpt_makeshortSIGlong_interp1d(shortSIG, longSIG):

    x = np.linspace(shortSIG[0], len(shortSIG), num=len(shortSIG), endpoint=True)
    y = shortSIG
    # print('x : ', x)


    # -------------
    kind = 'linear'
    # kind : Specifies the kind of interpolation as a string or as an integer specifying the order of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.

    if kind == 'linear':
        f = interp1d(x, y)
    elif kind == 'cubic':
        f = interp1d(x, y, kind='cubic')
    # -------------

    xnew = np.linspace(shortSIG[0], len(shortSIG), num=len(longSIG), endpoint=True)
    # print('xnew : ', xnew)

    siglong = f(xnew)

    return siglong

# -----------------------------------

def scale_feature_data(feat, plotORnot, type_scale):
    
    columns = ['0']
    dat = pd.DataFrame(data=feat, columns=columns)
    
    # which type of scaling
    if type_scale == 'minmax':
        # Values from 0 to 1
        scaled_data0 = minmax_scaling(dat, columns=columns)
        scaled_data = list(scaled_data0.to_numpy().ravel())
        
    elif type_scale == 'normalization':
        # normalization : same as mlxtend - Values from 0 to 1
        scaled_data = []
        for q in range(len(feat)):
            scaled_data.append( (feat[q] - np.min(feat))/(np.max(feat) - np.min(feat)) )
    
    elif type_scale == 'pos_normalization':
        # positive normalization : same as mlxtend - Values from 0 to 1
        shift_up = [i - np.min(feat) for i in feat]
        scaled_data = [q/np.max(shift_up) for q in shift_up]
    
    elif type_scale == 'standardization':
        # standardization : values are not restricted to a range, but scaled appropreately
        scaled_data = [(q - np.mean(feat))/np.std(feat) for q in feat]

    return scaled_data

# -----------------------------------

# level : the number of levels to decompose the time signal, le nombre des marquers par signale
def tsig_2_discrete_wavelet_transform(sig, waveletname, level, plotORnot):

    # On peut calculater dans deux façons: 0) dwt en boucle and then idwt, 1) wavedec et waverec
    # Mais le deux ne donnent pas le meme reponses, wavedec et waverec semble plus raisonable.
    coeff = wavedec(sig, waveletname, level)

    if plotORnot == 1:
        fig, axx = plt.subplots(nrows=len(coeff), ncols=1, figsize=(5,5))
        axx[0].set_title("coef")  # Pas certain si c'est coef0 ou coef1
        for k in range(len(coeff)):
            axx[k].plot(coeff[k], 'r') # output of the low pass filter (averaging filter) of the DWT
        plt.tight_layout()
        plt.show()

    return coeff

# -----------------------------------

def tsig_2_spectrogram(sig, fs, nfft, noverlap, img_dim, plotORnot):

    # -----------------------------------
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    # spectrum2D array : Columns are the periodograms of successive segments.
    # freqs1-D array : The frequencies corresponding to the rows in spectrum.
    # t1-D array : The times corresponding to midpoints of segments (i.e., the columns in spectrum).
    # imAxesImage : The image created by imshow containing the spectrogram.
    pxx, freqs, bins, img = ax.specgram(sig, nfft, fs, noverlap=noverlap)
    ax.axis('off')

    # -----------------------------------

    my_image = 'temp.png'
    fig.savefig(my_image)
    fname = os.path.abspath(os.getcwd()) + "/" +  my_image
    
    # Convert image to an array:
    # Read image 
    img = Image.open(fname)         # PIL: img is not in array form, it is a PIL.PngImagePlugin.PngImageFile 

    # -----------------------------------
    
    # Resize sectrogram image
    image = imgORmat_resize_imgORmat_CNN(img_dim, data_in=img, inpt='img3D', outpt='mat2D', norm='non', thresh='non')
    
    # -----------------------------------
    
    # Flatten image into a vector
    img_flatten = np.reshape(np.ravel(image), (img_dim*img_dim, ), order='F')
    # print('img_flatten.shape: ', img_flatten.shape)

    # -----------------------------------

    return img_flatten

# -----------------------------------

def tsig_2_continuous_wavelet_transform(sig, fs, scales, waveletname, img_dim, plotORnot):
    
    # e.g.:
    # scales = np.arange(1, 128)
    # waveletname = 'mexh'
    # sig = data2[0:100]
    # print('len(sig): ', len(sig))
    
    dt = 1/fs
    # print('dt : ', dt)

    coefficients, frequencies = pywt.cwt(sig, scales, waveletname, dt)
    coefficients = np.array(coefficients)
    ylen, xlen = coefficients.shape

    # Time by frequency plot of cwt : then flatten and use as a feature
    stop_val = len(sig)/fs
    x = np.arange(0, stop_val, dt)  # time
    y = frequencies # frequency 
    X, Y = np.meshgrid(x, y)
    Z = coefficients

    # Each cwt versus frequencies
    # for i in range(len(coefficients)):
    #     plt.plot(frequencies, coefficients[i])

    #  
    fig = plt.figure()
    ax = plt.axes() # creates a 3D axis by using the keyword projection='3d'
    ax.contourf(X, Y, Z, xlen, cmap=plt.cm.seismic) # contour fill
    ax.axis('off')

    # -----------------------------------

    my_image = 'temp.png'
    fig.savefig(my_image)
    fname = os.path.abspath(os.getcwd()) + "/" +  my_image

    # Convert image to an array:
    # Read image 
    img = Image.open(fname)  
    
    # -----------------------------------
    
    # Resize sectrogram image
    image = imgORmat_resize_imgORmat_CNN(img_dim, data_in=img, inpt='img3D', outpt='mat2D', norm='non', thresh='non')
    
    # -----------------------------------
    
    # Flatten image into a vector
    img_flatten = np.reshape(np.ravel(image), (img_dim*img_dim, ), order='F')
    # print('img_flatten.shape: ', img_flatten.shape)
    
    return img_flatten

# -----------------------------------

def resize_img(img, img_dim):
    if type(img) == 'numpy.ndarray':
        # img is an array, retuns an image object
        rgb_image = Image.fromarray(img , 'RGB')
    else:
        # img is an image object, returns an image object
        try:
            rgb_image = img.convert('RGB')
        except AttributeError:
            rgb_image = Image.fromarray(img , 'RGB')

    # Resize image into a 64, 64, 3
    new_h, new_w = int(img_dim), int(img_dim)
    img3 = rgb_image.resize((new_w, new_h), Image.ANTIALIAS)
    w_resized, h_resized = img3.size[0], img3.size[1]
    return img3

def convert_img_a_mat(img, outpt):
    mat = np.array(img)  # Convert image to an array
    if outpt == 'mat2D':
        # Transformer l'image de 3D à 2D
        # Convert image back to a 2D array
        matout = np.mean(mat, axis=2)
    elif outpt == 'img3D': # techniquement c'est un image parce qu'il y a trois RGB channels 
        matout = mat
    return matout

def norm_mat(mat2Dor3D, norm):
    if norm == 'zero2one':
        # Normalizer l'image entre 0 et 1
        norout = mat2Dor3D/255
    elif norm == 'negone2posone':
        # Normalize the images to [-1, 1]
        norout = (mat2Dor3D - 127.5) / 127.5
    elif norm == 'non':
        norout = mat2Dor3D
    return norout

def threshold_mat(mat2D, thresh):
    # Threshold image
    val = 255/2
    if thresh == 'zero_moins_que_val':
        row, col = mat2D.shape
        mat_thresh = mat2D
        min_val = np.min(mat_thresh)
        for i in range(row):
            for j in range(col):
                if mat_thresh[i,j] < val:
                    mat_thresh[i,j] = min_val
    elif thresh == 'non':
        mat_thresh = mat2D
    return mat_thresh

def imgORmat_resize_imgORmat_CNN(img_dim, data_in, inpt='img3D', outpt='mat2D', norm='non', thresh='non'):
    if inpt == 'img3D' and outpt=='mat2D':
        img = resize_img(data_in, img_dim)
        img3D = convert_img_a_mat(img, outpt)
        out = norm_mat(img3D, norm)
    elif inpt == 'mat2D' and outpt=='mat2D':
        data_in = np.array(data_in)
        img = Image.fromarray(data_in , 'L')
        img = resize_img(img, img_dim)
        mat2D = convert_img_a_mat(img, outpt)
        out = norm_mat(mat2D, norm)
    elif inpt == 'mat2D' and outpt=='img3D':
        data_in = np.array(data_in)
        img = Image.fromarray(data_in , 'L')
        img = resize_img(img, img_dim)
        img3D = convert_img_a_mat(img, outpt)
        out = norm_mat(img3D, norm)
    elif inpt == 'img3D' and outpt=='img3D':
        img = resize_img(data_in, img_dim)
        img3D = convert_img_a_mat(img, outpt)
        out = norm_mat(img3D, norm)

    return out

# -----------------------------------

def save_dat_pickle(outSIG, file_name="outSIG.pkl"):
    # Save data matrices to file
    open_file = open(file_name, "wb")
    pickle.dump(outSIG, open_file)
    open_file.close()

# -----------------------------------

def load_dat_pickle(file_name="outSIG.pkl"):
    open_file = open(file_name, "rb")
    dataout = pickle.load(open_file)
    open_file.close()
    return dataout

# -----------------------------------

def binarize_Y1Dvec_2_Ybin(Y):
    
    # Transform a 1D Y vector (n_samples by 1) to a Y_bin (n_samples by n_classes) vector

    # Ensure vector is of integers
    Y = [int(i) for i in Y]

    # Number of samples
    m_examples = len(Y)

    # Number of classes
    temp = np.unique(Y)
    unique_classes = [int(i) for i in temp]
    # print('unique_classes : ', unique_classes)

    whichone = 2
    # Binarize the output
    if whichone == 0:
        from sklearn.preprocessing import label_binarize
        Y_bin = label_binarize(Y, classes=unique_classes)  # seems to work now

    elif whichone == 1:
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        Y_bin = lb.fit_transform(Y)  # seems to work now
        
    elif whichone == 2:
        # By hand
        Y_bin = np.zeros((m_examples, len(unique_classes)))
        for i in range(0, m_examples):
            if Y[i] == unique_classes[0]:
                Y_bin[i,0] = 1
            elif Y[i] == unique_classes[1]:
                Y_bin[i,1] = 1
            elif Y[i] == unique_classes[2]:
                Y_bin[i,2] = 1
            elif Y[i] == unique_classes[3]:
                Y_bin[i,3] = 1
            elif Y[i] == unique_classes[4]:
                Y_bin[i,4] = 1
            elif Y[i] == unique_classes[5]:
                Y_bin[i,5] = 1
            elif Y[i] == unique_classes[6]:
                Y_bin[i,6] = 1
            elif Y[i] == unique_classes[7]:
                Y_bin[i,7] = 1
            elif Y[i] == unique_classes[8]:
                Y_bin[i,8] = 1
            elif Y[i] == unique_classes[9]:
                Y_bin[i,9] = 1
            elif Y[i] == unique_classes[10]:
                Y_bin[i,10] = 1
                
    print('shape of Y_bin : ', Y_bin.shape)

    return Y_bin, unique_classes

# -----------------------------------

def binarize_audio_signal(wind, data2):
    # wind = 1000
    a = int(np.floor(len(data2)/wind))
    vals = np.arange(0, a*wind, wind)
    sig = [np.max(data2[i:(i+wind)])*np.ones((wind)) for i in vals]    
    sig = make_a_properlist(sig)
    
    return sig

# -----------------------------------

def temporal_repeating_signatures(data2, timesteps, plotORnot):
    
    # ------------------------------------

    # choose wind such that sig is timesteps long
    size_of_mat = timesteps/2
    wind = int(len(data2)/size_of_mat)
    sig = binarize_audio_signal(wind, data2)

    min_border = np.mean(sig) + 1*np.std(sig)

    # On a besoin d'avoir au moins de 2 peaks
    peaks = []
    while len(peaks) < 2:
        # peaks, properties = find_peaks(sig, height=(min_border, np.max(sig)), prominence=100)
        peaks, properties = find_peaks(sig, height=(min_border, np.max(sig)))
        min_border = min_border - 10
    vv = [sig[i] for i in peaks]

    if plotORnot == 1:
        plt.plot(sig)
        plt.plot(peaks, vv, 'r*')
        plt.ylabel("Amplitude")
        plt.xlabel("Data points")
        plt.title("Bird call : binarized data2")
        plt.show()

    # ------------------------------------

    # Calculate the difference in peaks and find the difference that repeats the most
    # This repeating difference is the ideal window to cut the data for bird call temporal signatures
    peakdiff = [peaks[i+1]-peaks[i] for i in range(len(peaks)-1)]
    z = sorted(peakdiff)
    o = [len(list(str(z[i]))) for i in range(len(z))]

    from collections import Counter
    c = Counter(o)
    mc = c.most_common()[0][0]

    for i in range(len(z)):
        if len(list(str(z[i]))) == mc:
            wind = z[i]

    # ------------------------------------

    # Cut the data and compare each binarized piece, using a metric
    # To see which pieces are similar
    a = int(np.floor(len(sig)/wind))
    vals = np.arange(0, a*wind, wind)

    dic = {}
    dic2 = {}
    for i in vals:
        out = []
        out2 = []

        A = sig[i:(i+wind)]
        A = [int(r) for r in A]
        for j in vals:
            # import required libraries
            B = sig[j:(j+wind)]
            B = [int(r) for r in B]

            # cosine similarity
            #cosine = np.dot(A,B)/(norm(A)*norm(B))
            #out.append(cosine)

            # absolute error
            err = [np.abs(A[q]-B[q]) for q in range(len(A))]
            out.append(err)

            # height value
            height = np.max(A)
            out2.append(height)

        dic[i] = out
        dic2[i] = out2

    # ------------------------------------

    # tot = np.array(tot)
    # tot.shape
    # import seaborn as sns
    # sns.heatmap(data=tot, annot=True)

    # ------------------------------------

    # Take the mean across similarity measures for each temporal piece
    zz = [np.mean(dic[vals[i]]) for i in range(len(dic))]
    zz2 = [np.mean(dic2[vals[i]]) for i in range(len(dic2))]

    # ------------------------------------

    # Find the temporal piece that are most similar to all other pieces
    select_crit = 2

    if select_crit == 0:
        # No height restriction
        # ind_piece = np.argmax(zz)  # cosine similarity
        ind_piece = np.argmin(zz)  # absolute error
    elif select_crit == 1:
        # With height restriction
        # use height as threshold for zz
        marg = 1*np.std(sig)
        thresh = np.max(zz2) - marg
        print('thresh : ', thresh)
        # OU
        # thresh = np.min(zz2)
        store = [zz for i in range(len(zz)) if zz2[i] > thresh]

        # the select index using metric (cosine, absolute error)
        # ind_piece = np.argmax(store)  # cosine similarity
        ind_piece = np.argmin(store)  # absolute error
    elif select_crit == 2:
        # Rien a faire avec metrics, choisi par rapport la hauteur
        ind_piece = np.argmax(zz2)
    
    # Plot selected temporal piece, with data
    st = vals[ind_piece]
    endd = st+wind
    sig2 = sig[st:endd]

    if plotORnot == 1:
        plt.plot(sig)
        plt.plot(np.arange(st,endd), sig2)
        plt.ylabel("Amplitude")
        plt.xlabel("Data points")
        plt.title("Non-downsampled : Bird call and call signature")
        plt.show()

    # ------------------------------------

    # Downsampling
    v = downsample_sig(sig2, timesteps)

    # ------------------------------------
        
    return v

# -----------------------------------

def downsample_sig(sig, timesteps):

    # Downsampling
    a = int(np.floor(len(sig)/timesteps))
    vals = np.arange(0, a*timesteps, a)
    v = [sig[i:(i+a)][0] for i in vals] # duh, pick the first point from each window, that is why [0] is there
    
    return v

# -----------------------------------

# -----------------------------------
# LSTM
# -----------------------------------

def LSTM_arch(n_a, timesteps_train, feature, return_sequences, return_state, n_outputs):

    model = Sequential()
    
    model.add(LSTM(n_a, input_shape=(timesteps_train, feature), 
                   return_sequences=return_sequences, return_state=return_state))

    # Types of W initializer :
    initializer = tf.keras.initializers.HeUniform()

    if n_outputs == 2:
        model.add(Dense(n_outputs, activation='sigmoid', kernel_initializer=initializer))
    else:
        model.add(Dense(n_outputs, activation='softmax', kernel_initializer=initializer))

    # Compile the model for training
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # opt = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    # opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

    # Si vous utilisez softmax activation, la taille de sortie est plus grand que deux donc il faut categorical_crossentropy
    if n_outputs == 2:
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'
    else:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'

    # model.summary()

    return model

# -----------------------------------

def create_X_batch_timesteps_feature(X, y_bin, input_width, return_sequences):
    
    g = np.argmax(y_bin, axis=1)
    n_outputs = len(np.unique(g))
    print('n_outputs:' , n_outputs)
    
    batch = int(np.floor(X.shape[0]/input_width))

    Xout = []
    yout = []
    for i in range(batch):
        st = i*input_width
        endd = st + input_width
        Xout.append(X[st:endd,:])
        yout.append(y_bin[st:endd,:])

    Xout = np.array(Xout)
    yout = np.array(yout)

    batch, timesteps, feature = Xout.shape
    print('batch:' , batch)
    print('timesteps:' , timesteps)
    print('feature:' , feature)
    
    # ---------------------
    # This part folds y for the return_sequences specification:
    # So you do not need to remember how to fold y
    # ---------------------
    # if return_sequences == True:
        # yout : (batch, timesteps, n_outputs)
    if return_sequences == False:
        # yout : (batch, 1, n_outputs)
        yt = [yout[i,0:1,:] for i in range(batch)]
        yout = np.array(yt)
        yout = np.reshape(yout, (batch, n_outputs))
    # ---------------------
    
    print('Xout:' , Xout.shape)
    print('yout:' , yout.shape)  
        
    tf_data = tf.data.Dataset.from_tensor_slices((Xout, yout))
    tf_data = tf_data.batch(batch)

    return tf_data, Xout, yout, n_outputs

# -----------------------------------

# -----------------------------------
# Transformer
# -----------------------------------
def reshapeY3D_a_Y1D(Y_bin):
    # Y_bin : (batch, timesteps, n_outputs) OU Y_bin : (batch, n_outputs)
    
    batch = Y_bin.shape[0]
    
    if len(Y_bin.shape) == 2:
        # Y_bin : (batch, n_outputs)
        temp = [np.argmax(Y_bin[i,:]) for i in range(batch)]   # (batch,)
    
    elif len(Y_bin.shape) == 3:
        # Y_bin : (batch, timesteps, n_outputs)
        temp = [np.argmax(Y_bin[i,0:1,:]) for i in range(batch)]   # (batch,)
    
    Y = np.array(temp)
    Y = np.reshape(Y, (batch,)) # Y : (batch, )
    
    return Y

# -----------------------------------

def run_Transformer(X_train, X_test, Y_train, Y_test, patience, batch_size, epochs):
    
    # X_train : (batch_train, timesteps_train, feature)
    # X_test : (batch_test, timesteps_test, feature)
    
    # Y_train : (batch_train, n_outputs)
    # Y_test : (batch_test, n_outputs)
    
    # -------------------------------
    
    batch_train = X_train.shape[0]
    timesteps_train = X_train.shape[1]
    feature = X_train.shape[2]
    
    batch_test = X_test.shape[0]
    timesteps_test = X_test.shape[1]
    
    n_outputs = Y_train.shape[1]
    
    # -------------------------------
    
    # Model architecture
    
    # Encoder (shuffling and transformation of the data)
    tf.random.set_seed(1)  #10
    
    # unique words, or unique data points to a certain precision
    precision = 0.1
    input_vocab_size=len(np.arange(0,1,precision))
    
    # the maximum number of words in each sentence, OR timesteps
    maximum_position_encoding = timesteps_train
    
    encoderq = Encoder(num_layers=6, embedding_dim=feature, num_heads=1, fully_connected_dim=2*feature, 
                       input_vocab_size=input_vocab_size,
                       maximum_position_encoding=maximum_position_encoding, dropout_rate=0.1, 
                       layernorm_eps=1e-6, textORnot='timeseries')
    
    training = True  # training for Dropout layers
    mask = None #create_padding_mask(X_train)  # create_look_ahead_mask(x.shape[1]) # None
    encoder_X_train = encoderq(X_train, training, mask)   # output is shape=(batch_train, timesteps_train, feature)
    
    mask = None #create_padding_mask(X_test)  # create_look_ahead_mask(x.shape[1]) # None
    encoder_X_test = encoderq(X_test, training, mask)   # output is shape=(batch_test, timesteps_test, feature)
    
    # -------------------------------
    
    # Final Fully Connected
    model = Sequential()
    initializer = tf.keras.initializers.HeUniform()
    if n_outputs > 2:
        model.add(Dense(n_outputs, activation='softmax', kernel_initializer=initializer))
    else:
        model.add(Dense(n_outputs, activation='sigmoid', kernel_initializer=initializer))

    # --------

    # Compile the model for training
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    if n_outputs > 2:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, mode='min')

    # -------------------------------
    
    X_train_1D = np.reshape(encoder_X_train, (batch_train, timesteps_train*feature))
    X_test_1D = np.reshape(encoder_X_test, (batch_test, timesteps_test*feature))
    
    verbose = 2
    #history = model.fit(X_train_1D, Y_train, epochs=epochs, 
    #                    validation_data=(X_test_1D, Y_test), 
    #                    batch_size=batch_size, callbacks=[early_stopping], verbose=verbose)
    history = model.fit(X_train_1D, Y_train, epochs=epochs, 
                        validation_data=(X_test_1D, Y_test), 
                        batch_size=batch_size, verbose=verbose)
    history_df = pd.DataFrame(history.history)
    
    # -------------------------------
    
    return model, history_df

# -----------------------------------

# -----------------------------------
# N-Layer NN
# -----------------------------------

def dcgan_arch(n_outputs, img0, den_activation):
    
    hidden_dim = 128
    
    model = Sequential()
    model.add(Dense(hidden_dim * 4, activation=den_activation, input_shape=(img0, )))
    model.add(Dense(hidden_dim * 2, activation=den_activation))
    model.add(Dense(hidden_dim, activation=den_activation))
    
    initializer = tf.keras.initializers.HeUniform()
    # initializer = tf.keras.initializers.HeNormal()
    # initializer = tf.keras.initializers.GlorotUniform()
    if n_outputs > 2:
        model.add(Dense(n_outputs, activation='softmax', kernel_initializer=initializer))
    else:
        model.add(Dense(n_outputs, activation='sigmoid', kernel_initializer=initializer))

    # --------
    
    # Compile the model for training
    opt = keras.optimizers.Adam()
    # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    if n_outputs > 2:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'
    
    # model.summary()
    
    return model


# -----------------------------------

# -----------------------------------
# CNN
# -----------------------------------

# MaxPooling Convolutional 2D repeating layers

def MPCNN_arch(n_outputs, img_dim, rgb_layers, mod):
    
    # Typical architecture MPCNN architecture using alternating convolutional and max-pooling layers. 
    
    model = Sequential()  # initialize Sequential model
    
    if mod == 0:
        model.add(Conv2D(32, (5,5), strides=(1,1), padding='same', input_shape=(img_dim, img_dim, rgb_layers)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        model.add(Conv2D(32 * 2, (5,5), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        model.add(Flatten())
    elif mod == 1:
        model.add(Conv2D(8,(4,4), strides=(1,1), padding='same', input_shape=(img_dim, img_dim, rgb_layers)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((8,8), strides=(8,8), padding='same'))
        model.add(Conv2D(16,(2,2), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((4,4), strides=(4,4), padding='same'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
    elif mod == 2:
        # 2D Convolutional model using MFCC
        model.add(Conv2D(32, (4,10), padding="same", input_shape=(img_dim, img_dim, rgb_layers)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,1), strides=(1,1), padding='same'))
        
        model.add(Conv2D(32, (4,10), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,1), strides=(1,1), padding='same'))

        model.add(Conv2D(32, (4,10), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,1), strides=(1,1), padding='same'))

        model.add(Conv2D(32, (4,10), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1,1), strides=(1,1), padding='same'))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    elif mod == 3:
        # 1D Convolutional model using MFCC
        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)

        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)
        out = Dense(nclass, activation=softmax)(x)

    initializer = tf.keras.initializers.HeUniform()
    # initializer = tf.keras.initializers.HeNormal()
    # initializer = tf.keras.initializers.GlorotUniform()
    if n_outputs > 2:
        model.add(Dense(n_outputs, activation='softmax', kernel_initializer=initializer))
    else:
        model.add(Dense(n_outputs, activation='sigmoid', kernel_initializer=initializer))
    
    print('model.output_shape :', model.output_shape)
    # model.output_shape : (None, 1)
    # --------
    
    # Compile the model for training
    opt = keras.optimizers.Adam()
    # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    if n_outputs > 2:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'
    
    # model.summary()
    
    return model

# -----------------------------------

def encoderdecoder_arch(n_outputs, img_dim, rgb_layers):
    
    base_dimension = 64          
    
    model = Sequential()
    # 1ère valeur (filters) : le nombre de tranches "(kernel_val,kernel_val)" qui composent l'image de sortie
    # 2eme valeur (kernel_size) : la taille de la carre/filtre que on glisse au dessous l'image 
    # 3eme valeur (stride): Le plus grande le stride valeur le plus petite l'image sortie : on prends z_dim/stride_num
    
    # --------
    # Entrée = (img_dim, img_dim, 1)
    model.add(Conv2D(base_dimension, (5,5), strides=(2,2), padding='same', input_shape=(img_dim, img_dim, rgb_layers)))
    print('model.output_shape :', model.output_shape)
    # Sortie = 
    # taille_sortie = (28 + 2*p - 5)/2 + 1

    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    # --------

    # --------
    # Entrée = 
    model.add(Conv2D(base_dimension * 2, (5,5), strides=(2,2), padding='same'))
    print('model.output_shape :', model.output_shape)
    # Sortie = 

    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    # --------

    # --------
    model.add(Flatten())

    print('model.output_shape :', model.output_shape)
    # model.output_shape : (None, 4096)
    # --------
    
    initializer = tf.keras.initializers.HeUniform()
    # initializer = tf.keras.initializers.HeNormal()
    # initializer = tf.keras.initializers.GlorotUniform()
    if n_outputs > 2:
        model.add(Dense(n_outputs, activation='softmax', kernel_initializer=initializer))
    else:
        model.add(Dense(n_outputs, activation='sigmoid', kernel_initializer=initializer))
    
    print('model.output_shape :', model.output_shape)
    # model.output_shape : (None, 1)
    # --------
    
    # Compile the model for training
    opt = keras.optimizers.Adam()
    # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    if n_outputs > 2:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])  # optimizer='adam'
    
    # model.summary()
    
    return model

# -----------------------------------

def initialize_CNN(X_train, X_test, Y_train, Y_test, img_dim): 
    
    # X_train: (batch_train, timesteps_train, feature_train)
    # Y_train: (batch_train, timesteps_train, n_outputs)
    # X_test: (batch_test, timesteps_test, feature_train)
    # Y_test: (batch_test, timesteps_test, n_outputs)
    
    batch_train = X_train.shape[0]
    batch_test = X_test.shape[0]
    
    # ----------------
    
    # Tranformez X(batch, timestamps, feature) into X(batch, img_dim, img_dim, 3)
    X_train_img = Xbtf_2_Xbii3(X_train, img_dim, batch_train)
    X_test_img = Xbtf_2_Xbii3(X_test, img_dim, batch_test)
    
    # ----------------
    
    # Enlevez timesteps
    if len(Y_train.shape) == 3:
        Y_train =  [Y_train[i,0:1,:] for i in range(batch_train)]
        Y_test =  [Y_test[i,0:1,:] for i in range(batch_test)]

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    # ----------------
    
    # shape of X_train_img :  (batch_train, img_dim, img_dim, 3)
    # shape of Y_train :  (batch_train, n_outputs)
    # shape of X_test_img :  (batch_test, img_dim, img_dim, 3)
    # shape of Y_test :  (batch_test, n_outputs)
    
    X_train_img = np.asarray(X_train_img, dtype = np.float16, order ='C')  # np.float16, np.float32, np.float64
    Y_train = np.asarray(Y_train, dtype = np.float16, order ='C')
    X_test_img = np.asarray(X_test_img, dtype = np.float16, order ='C')
    Y_test = np.asarray(Y_test, dtype = np.float16, order ='C')
    
    print('X_train_img:' , X_train_img.shape)
    print('Y_train:' , Y_train.shape)
    print('X_test_img:' , X_test_img.shape)
    print('Y_test:' , Y_test.shape)
    
    return X_train_img, X_test_img, Y_train, Y_test
    
    
# Tranformez X(batch, timestamps, feature) into X(batch, img_dim, img_dim, 3)
def Xbtf_2_Xbii3(X, img_dim, batch):
    X_img = []
    for i in range(batch):
        X_1D = X[i,:,:].flatten()
        
        if i == 0:
            n = int(np.floor(np.sqrt(len(X_1D))))
    
        # fold into a square
        mat = np.reshape(X_1D[0:n*n], (n, n))
    
        image = imgORmat_resize_imgORmat_CNN(img_dim, mat, inpt='mat2D', outpt='img3D', norm='non', thresh='non')
        
        X_img.append(image)
    
    X_img = np.array(X_img)
    
    return X_img

# -----------------------------------

def run_CNN(X_train, X_test, Y_train, Y_test, patience, batch_size, epochs):
    
    # X_train: (batch_train, timesteps_train, feature_train)
    # Y_train: (batch_train, timesteps_train, n_outputs)
    # X_test: (batch_test, timesteps_test, feature_train)
    # Y_test: (batch_test, timesteps_test, n_outputs)
    
    # ----------------
    img_dim = 64
    
    # Folding data into CNN image format:
    X_train_img, X_test_img, Y_train, Y_test  = initialize_CNN(X_train, X_test, Y_train, Y_test, img_dim)
    
    # shape of X_train_img :  (batch_train, img_dim, img_dim, 3)
    # shape of Y_train :  (batch_train, n_outputs)
    # shape of X_test_img :  (batch_test, img_dim, img_dim, 3)
    # shape of Y_test :  (batch_test, n_outputs)
    
    n_outputs = Y_test.shape[1]
    
    # ----------------
    
    # Model architecture
    epochs = 100
    batch_size = 32
    patience = 5
    
    tot = []
    tot_mod = []
    mod_type = ['mpcnn', 'encdec'] # CNN model architecture type
    rgb_layers = X_train_img.shape[3]
    
    for i in range(2):
        if i == 0:
            mod = 2 # 0, 1, 2, 3=1D
            model = MPCNN_arch(n_outputs, img_dim, rgb_layers, mod)
        elif i == 1:
            model = encoderdecoder_arch(n_outputs, img_dim, rgb_layers)
            
    
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, mode='min')
        
        # -------------------------------
        
        history = model.fit(X_train_img, Y_train, epochs=epochs, validation_data=(X_test_img, Y_test), batch_size=batch_size, callbacks=[early_stopping], verbose=2)
        
        history_df = pd.DataFrame(history.history)
        
        history_df.loc[:, ['loss', 'val_loss']].plot();
        print("Max validation loss: {}".format(history_df['val_loss'].max()))
        
        history_df.loc[:, ['accuracy', 'val_accuracy']].plot();
        print("Max validation accuracy: {}".format(history_df['val_accuracy'].max()))
        
        out = [history_df.iloc[:,i].mean() for i in range(len(history_df.columns))]
    
        tot.append(out)
        tot_mod.append(model)
    
    tot = np.array(tot)
    
    # -------------------------------
    
    a = np.argmax(tot[:,1])  # train
    b = np.argmax(tot[:,6])  # test
    suf = ['train', 'test']
    tr_noms = ['loss_', 'acc_', 'prec_', 'recall_', 'roc_auc_']

    list2 = [j+i for i in suf for j in tr_noms]
    list2

    dict_out = {}
    for i in range(len(list2)):
        if i < len(list2)/2:
            r = tot[a,i]
        else:
            r = tot[b,i]
        dict_out[list2[i]] = r


    # ajoutez au dictionaire
    dict_out['mod_train'] = mod_type[a]
    dict_out['mod_test'] = mod_type[b]

    # -------------------------------

    cnn2D_model_best = tot_mod[b]
    
    # -------------------------------
    
    return cnn2D_model_best, dict_out

# -----------------------------------




# -----------------------------------



# -----------------------------------



# -----------------------------------
