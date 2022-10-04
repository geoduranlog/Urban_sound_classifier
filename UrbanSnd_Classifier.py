# -*- coding: utf-8 -*-
"""
#=========Model to Classify Urban Sounds=====
#09.08.2022

SUMMARY:
Urban Sound classification, audio labeled data from 
https://urbansounddataset.weebly.com/urbansound8k.html

* classID:
A numeric identifier of the sound class:
0 = air_conditioner
1 = car_horn
2 = children_playing
3 = dog_bark
4 = drilling
5 = engine_idling
6 = gun_shot
7 = jackhammer
8 = siren
9 = street_music

"""


file_path='/Users/alejandro/Survey_Platform/Baby_App/Work/Sound_Classifier_DL/'
#cd file_path

# ==Clear everything (~like a new shell) - useful in Spyder
from IPython import get_ipython
get_ipython().magic('reset -sf')



#--- Calculate total running time
import time
start_time = time.time()

#To install librosa
# python -m pip install librosa (package for music and audio analysis)

# ====Import Libraries, Modules=====
#import IPython.display as ipd
from playsound import playsound
import librosa
import librosa.display
import matplotlib.pyplot as plt

import pandas as pd
import os
import numpy as np
  

#path= '/Users/alejandro/Survey_Platform/Baby_App/Work/Sound_Classifier_DL'


#-- Check patterns in different sound types
#Class 6: Gun shot
filename = 'UrbanSound8K/audio/fold1/7061-6-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveshow(data,sr=sample_rate)  #Use '_' to save the waveshow object 
#ipd.Audio(filename) #Play audio in Jupter
#playsound(filename)

# Class 2: children_playing
filename = 'UrbanSound8K/audio/fold1/15564-2-0-2.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
librosa.display.waveshow(data,sr=sample_rate)  
#playsound(filename)

#Class 1:Car horn
filename = 'UrbanSound8K/audio/fold1/24074-1-0-9.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
librosa.display.waveshow(data,sr=sample_rate)  
playsound(filename)

#--- Load Metadata (Audio labels)
metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
metadata.head()
metadata.columns


#-Temp select only fold 1 (using less data)
#metadata=metadata[metadata['fold']==1]


#Class distribution
metadata['class'].value_counts()

#A predifined class in folder helper
from extra.wavfilehelper import WavFileHelper 
wavfilehelper = WavFileHelper()

audiodata = []
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath('/Users/alejandro/Survey_Platform/Baby_App/Work/Sound_Classifier_DL/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))  
    #file_name = os.path.join(os.path.abspath('/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    data = wavfilehelper.read_file_properties(file_name)
    audiodata.append(data)

# Convert into a Panda dataframe
audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])


#---- See data proportions in nb chanels, sample rates and bit depth--
#to see percentages -> df.value_counts(normalize=True)

# Proportion (%) of num of channels present (2:estereo,  1:mono)
print(audiodf.num_channels.value_counts(normalize=True))

Class_count=audiodf['num_channels'].value_counts().to_frame()
fig = plt.figure()
plt.pie(Class_count['num_channels'], labels =['Estereo','Mono'],
#plt.pie(Class_count['num_channels'], labels =[Class_count['num_channels'].index[0],Class_count['num_channels'].index[1]],
        autopct='%1.1f%%',explode = [.02,0])
plt.title('Distribution of Channels')
plt.show()


# sample rates 
print(audiodf.sample_rate.value_counts(normalize=True))
audiodf.sample_rate.value_counts().sort_index().plot(kind='bar') # pie  bar   #sort_values()
plt.title('Distribution of Sampling rates')
plt.xlabel('Sample rate (Hz)')
plt.ylabel('Nb instances')




# bit depth
print(audiodf.bit_depth.value_counts(normalize=True))

audiodf.bit_depth.value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Amplitude Dinamic Range')
plt.xlabel('Bith Depth')



"""
==============================================
                Preprocessing
==============================================

The following audio properties need preprocessing to ensure consistency 
across the whole dataset:

Audio Channels
Sample rate
Bit-depth

"""

#-- Comparing sample rates (librosa converts it to 22.05 KHz by default)
from scipy.io import wavfile as wav
librosa_audio, librosa_sample_rate = librosa.load(filename) 
scipy_sample_rate, scipy_audio = wav.read(filename) 

print('Original sample rate:', scipy_sample_rate) 
print('Librosa sample rate:', librosa_sample_rate) 

#Bith-depth (Dynamic Range)
#Librosa normalise the data by default: -1 and 1. This removes the complication
# of the dataset having a wide range of bit-depths.
print('Original audio file min~max range:', np.min(scipy_audio), 'to', np.max(scipy_audio))
print('Librosa audio file min~max range:', np.min(librosa_audio), 'to', np.max(librosa_audio))



#-- Librosa converts the signal to mono--
# Plot original audio (with 2 channels)
plt.figure(figsize=(12, 4))
plt.plot(scipy_audio)

# Librosa audio with channels merged 
plt.figure(figsize=(12, 4))
plt.plot(librosa_audio)




"""
==============================================
                Feature Extraction
==============================================

Using Mel-Frequency Cepstral Coefficients (MFCC) 

The MFCC summarises the frequency distribution across the window size, 
then it is possible to analyse both the frequency and time characteristics.
These audio representations allow to identify features for classification.

"""

mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape) # librosa calculated a series of 40 MFCCs over 173 frames (Hyp: windows).

#Plot spectogram
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time') #y_axis='mel'

#https://librosa.org/doc/main/generated/librosa.display.specshow.html


# y=librosa_audio
# sr=librosa_sample_rate
# hop_length = 1024

# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

# D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
#                             ref=np.max)
# librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
#                          x_axis='time', ax=ax[1])
# ax[1].set(title='Log-frequency power spectrogram')
# ax[1].label_outer()
# fig.colorbar(img, ax=ax, format="%+2.f dB")





#------

S=librosa.feature.melspectrogram(y=librosa_audio, sr=librosa_sample_rate, n_mels=40,
                                  fmax=8000)

#--Figure
fig, ax = plt.subplots(nrows=2, sharex=True)
#Mel Spectrogram
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()

#MFCC
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')

# --Display a spectrogram using librosa.display.specshow--
Xsound = librosa.stft(librosa_audio)
Xdb = librosa.amplitude_to_db(abs(Xsound))
#plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=librosa_sample_rate, x_axis='time', y_axis='hz')



# m_slaney = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40,dct_type=2)
# m_htk = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40, dct_type=3)
# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
# img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax[0])
# ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')
# fig.colorbar(img, ax=[ax[0]])
# img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
# ax[1].set(title='HTK-style (dct_type=3)')
# fig.colorbar(img2, ax=[ax[1]])
#------


# Extract  MFCCs for each audio file in the dataset and store it in a df
# along with it's classification label.

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled




fulldatasetpath = '/Users/alejandro/Survey_Platform/Baby_App/Work/Sound_Classifier_DL/UrbanSound8K/audio/'     #audio_fold1  audio
features = []

#"""
#---- Iterate through each sound file and extract the features--- 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["class"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
print('Finished feature extraction from ', len(featuresdf), ' files') 
#"""

#------- SAVE features df ------
##featuresdf.to_csv(file_path+'MFCC_features.csv')
#featuresdf.to_pickle("MFCC_features.pkl")



# #------- LOAD features df ------
# #since I saved it with indices, I need to load only the relevant info (feature, label)
# featuresdf= pd.read_csv('MFCC_features_10folds.csv',usecols=['feature', 'class_label']) #   [1,2] # 

# featuresdf.dtypes # Problem, all load data is taken as char (object) including '\n'
# featuresdf['feature'][0]

# #featuresdf['features'] = featuresdf['feature'].str.replace('\n', '').astype(float)
# featuresdf['feature'] = featuresdf['feature'].str.replace('\n', '')


# a=featuresdf2['feature']

# a2=a[0]


# #----- Load Features .pkl ---------
#featuresdf2=pd.read_pickle("MFCC_features_10folds_2ndround.pkl")
#featuresdf = featuresdf2

#-- Nan Values? --
#featuresdf.isnull().values.any()
featuresdf.isnull().sum()


# Nb of features per each sound, all should be = 40
# i=0
# count=0
# for i in range(0,len(featuresdf)):
#     if len(featuresdf.feature[i])!= 40:
#         print ( len(featuresdf.feature[i]) )
        


#-----------------------------------------------------------------------------
#----Encode the categorical text data into model-understandable numerical data

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())   # X[1000:1074]  np.size(featuresdf.iloc[1000][0])
y = np.array(featuresdf.class_label.tolist())

#----- Encode the classification labels
# Use tf.keras.utils.to_categorical(arr, num_classes=None, dtype='float32')
# It converts a class vector (integers) to binary class matrix.
# E.g. for use with categorical_crossentropy
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))   ## yy[1000:1074]  Matrix of nb_files x nb_categories


# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)



"""
==============================================
                Computing the Model - Model 1
==============================================

Building a fully connected (i.e., dense) sequencial NN model

Model architecture:
    Input layer: 40 input nodes (one per each MFCC  coefficient, i.e. column , i.e X) -> 40 = np.size(X[1000]) = np.size(featuresdf.iloc[1000][0])
    2 Hidden layers: 256 nodes each with a ReLU activation function. 50% Dropout value.
    Output layer: 10 nodes , one per each audio label (possible classification)
                  with Softmax activation function (~probabilities [0,1])                 
    

Notes:
Dropout value of 50%. This will randomly exclude nodes from each update cycle 
which in turn results in a network that is capable of better generalisation 
and is less likely to overfit the training data.

Softmax makes the output sum up to 1 so the output can be interpreted as probabilities.
The model will then make its prediction based on which option has the highest probability.


"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import np_utils
from sklearn import metrics 

num_labels = yy.shape[1]
#filter_size = 2


#===============================================
#          Building the model
#===============================================
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()


#--- Compile model  ---
#loss: categorical_crossentropy. This is the most common choice for classification.
# A lower score indicates that the model is performing better.

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 


# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


#===============================================
#          Training the model
#===============================================

#Start with a low batch size, as having a large batch size can reduce 
#the generalisation ability of the model

from tensorflow.keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 200
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_mlp.hdf5', 
                               verbose=1, save_best_only=True)

# Measure trainig time
start = datetime.now()

# Training
#model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
history = model.fit(x_train, y_train, epochs=num_epochs, 
        batch_size=num_batch_size, 
        validation_data=(x_test, y_test),
        callbacks=[checkpointer], verbose=1
) #validation_split = 0.2


duration = datetime.now() - start
print("Training completed in time: ", duration)



#-- Figure Overfitting control --
epochs = range(1, num_epochs+1)

plt.rcParams['text.color'] = 'black'
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(epochs, history.history["loss"], label="train")
plt.plot(epochs, history.history["val_loss"], label="test")
plt.legend(loc='upper right')

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.plot(epochs, history.history["accuracy"], label="train")
plt.plot(epochs, history.history["val_accuracy"], label="test")
plt.legend(loc='lower right')

plt.show()







#====== Load saved model =====
#import tensorflow as tf
#model = tf.keras.models.load_model('saved_models/weights_allFolds12082022.best.basic_mlp.hdf5')

#===============================================
#          Testing the model
#===============================================
# Evaluating the model on the training and testing set
score_train = model.evaluate(x_train, y_train, verbose=0)  # = [loss, acc]
print("Training Accuracy: ", score_train[1])

score_test = model.evaluate(x_test, y_test)
print("Testing Accuracy: ", score_test[1])

# If there is no large diff  between Tes_Acc and Train_Acc => no overfitting
Diff= ( np.abs(score_test[1]-score_train[1])/max(score_test[1],score_train[1]) )*100  #Percentual diff


#-- Predictions---
#Build a method to test  models predictions on a specified audio .wav file.


def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    return np.array([mfccsscaled])




def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 

    predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba_vector = model.predict(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )



#----- Validation/QC 1 - sanity check-----
# Using same files in the exploratory part to predict their category


#Class 6: Gun shot -- (wrong prediction with small dataset, good with large Dset)
filename = 'UrbanSound8K/audio/fold1/7061-6-0-0.wav'
print_prediction(filename)

# Class 2: children_playing
filename = 'UrbanSound8K/audio/fold1/15564-2-0-2.wav'
print_prediction(filename)

#Class 1:Car horn
filename = 'UrbanSound8K/audio/fold1/24074-1-0-9.wav'
print_prediction(filename)

#Class 3:Dog Bark
filename = 'UrbanSound8K/audio/fold1/9031-3-2-0.wav'
print_prediction(filename)
#playsound(filename)


#----- Validation/QC 2 - External audio-----
#New audio that weren't part of either test or training data

# External audio -  Dog
filename = 'UrbanSound8K/eval_audio/Evaluation_audio_dog_bark_1.wav'
print_prediction(filename)
#playsound(filename)

# External audio -  Drilling (wrong pred?)
filename = 'UrbanSound8K/eval_audio/Evaluation_audio_drilling_1.wav'
print_prediction(filename)
#playsound(filename)

# External audio -  Gun shot (wrong pred)
filename = 'UrbanSound8K/eval_audio/Evaluation_audio_gun_shot_1.wav'
print_prediction(filename)
#playsound(filename)

# External audio -  Siren
filename = 'UrbanSound8K/eval_audio/Evaluation_audio_siren_1.wav'
print_prediction(filename)
playsound(filename)



# #----- Evaluation Metrics-----
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix


predicted_vector = np.argmax(model.predict(x_test), axis=-1)
y_pred = le.inverse_transform(predicted_vector) 
y_test_label = le.inverse_transform(np.argmax(y_test, axis=1))


import seaborn as sns
#Since I'm using Softmax the result is a prob vector
# Then, use np.argmax to get the most probable class
y_pred2 =  np.argmax(model.predict(x_test), axis=-1)
cf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred2)
#cf_matrix=confusion_matrix(y_test_label, y_pred)


#----- Model Performance with test Data----
print(classification_report(np.argmax(y_test, axis=1), y_pred2))

score_test = model.evaluate(x_test, y_test)
print("Testing Accuracy: ", score_test[1])



labels =['air_conditioner','car_horn','children_playing','dog_bark','drilling',
         'engine_idling','gun_shot','jackhammer','siren','street_music']


# #sns.heatmap(cf_matrix, annot=True,fmt='.2%', cmap='coolwarm') #Blues coolwarm rocket
# sns.heatmap(cf_matrix, cmap='coolwarm') 
# plt.ylabel('True label')
# plt.xlabel('Predicted label')


ax = sns.heatmap(cf_matrix, annot=True, fmt='g');
## Modify the Axes Object directly to set various attributes such as the
## Title, X/Y Labels.
ax.set_title('Clasification Report Urban sound');
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label');
## For the Tick Labels, the labels should be in Alphabetical order
#ax.xaxis.set_ticklabels(labels)
#ax.yaxis.set_ticklabels(labels)






#---- Confution Matrix Plot function ---
#https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Clasification Report Urban sound');
    # Note that due to returning the created figure object, when this funciton is called in a notebook
    # the figure willl be printed twice. To prevent this, either append ; to your function call, or
    # modify the function by commenting out this return expression.
    #return fig

#-- Confusion matrix figure 
print_confusion_matrix(cf_matrix, labels, figsize = (10,7), fontsize=14)


"""
===============================================================================
                Computing the Model - Model 2
===============================================================================

Building a CNN sequencial model

Model architecture:
    4 Conv2D layers. Each layer will increase in size (i.e. Nb output filters) from 16, 32, 64 to 128 
                    The kernel size is 2x2
    1 final dense layer with 10 output nodes (one for each possible sound classification) 



"""



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import np_utils
from sklearn import metrics 



#Before, MFCC vectors would vary in size for the different audio files (depending on the samples duration).
#However, CNNs require a fixed size for all inputs. To overcome this we will zero pad the 
#output vectors to make them all the same size.
max_pad_len = 174

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs



#--- Load Metadata (Audio labels)
metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')


# #-Temp select only fold 1 (using less data)
#metadata=metadata[metadata['fold']==1]


#-- Extract features - Long ~20min
features = []
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath('/Users/alejandro/Survey_Platform/Baby_App/Work/Sound_Classifier_DL/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))  
    class_label = row["class"]
    data = extract_features(file_name)
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf_cnn = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf_cnn), ' files') 

#------- SAVE features df ------
#featuresdf_cnn.to_pickle("MFCC_features_10folds_cnn.pkl")


# #----- Load Features .pkl ---------
#featuresdf_cnn=pd.read_pickle("MFCC_features_10folds_cnn.pkl")



#--- Encode Categorical variables --
# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf_cnn.feature.tolist())
y = np.array(featuresdf_cnn.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 



#--- Split into test and train --
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


num_rows = 40
num_columns = 174 # = max_pad_len
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

#=========== CNN Model ===========
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)



#===============================================
#          Training the CNN model
#===============================================
#training a CNN can take time, start with a low number of epochs and batch size.
# If the model is converging, you can increase both numbers.

num_epochs = 72
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)

# Measure trainig time
start = datetime.now()

# ---- Training ----- Long ~ 45min
#model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
history = model.fit(x_train, y_train, epochs=num_epochs, 
        batch_size=num_batch_size, 
        validation_data=(x_test, y_test),
        callbacks=[checkpointer], verbose=1
) #validation_split = 0.2


duration = datetime.now() - start
print("Training completed in time: ", duration)


# # ---- Load saved model, i.e. weigths -----
# checkpoint_path = 'saved_models/weights.best.basic_cnn.hdf5'
# model.load_weights(checkpoint_path)


# # --- Impot you can also use this to save and load models---
# from sklearn.externals import joblib
# filename = "Completed_model.joblib"
# joblib.dump(model, filename) # To save
# loaded_model = joblib.load(filename)  # To load


#-- Figure Overfitting control --
epochs = range(1, num_epochs+1)

plt.rcParams['text.color'] = 'black'
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(epochs, history.history["loss"], label="train")
plt.plot(epochs, history.history["val_loss"], label="test")
plt.legend(loc='upper right')
#plt.xlim(40 ,70)
#plt.ylim(0.3 ,0.5)
plt.ylim(0 ,1)


plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.plot(epochs, history.history["accuracy"], label="train")
plt.plot(epochs, history.history["val_accuracy"], label="test")
plt.legend(loc='lower right')

plt.show()


#===============================================
#          Testing the model
#===============================================
# Evaluating the model on the training and testing set
score_train = model.evaluate(x_train, y_train, verbose=0)  # = [loss, acc]
print("Training Accuracy: ", score_train[1])

score_test = model.evaluate(x_test, y_test)
print("Testing Accuracy: ", score_test[1])

# If there is no large diff  between Tes_Acc and Train_Acc => no overfitting
Diff= ( np.abs(score_test[1]-score_train[1])/max(score_test[1],score_train[1]) )*100  #Percentual diff


#-- Predictions---
#Build a method to test  models predictions on a specified audio .wav file. Adapted for CNN shape

def print_prediction(file_name):
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    #predicted_vector = model.predict_classes(prediction_feature)
    predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )




#----- Validation/QC 1 - sanity check-----
# Using same files in the exploratory part to predict their category


#Class 6: Gun shot -- (wrong prediction with small dataset, good with large Dset)
filename = 'UrbanSound8K/audio/fold1/7061-6-0-0.wav'
print_prediction(filename)

# Class 2: children_playing
filename = 'UrbanSound8K/audio/fold1/15564-2-0-2.wav'
print_prediction(filename)

#Class 1:Car horn
filename = 'UrbanSound8K/audio/fold1/24074-1-0-9.wav'
print_prediction(filename)

#Class 3:Dog Bark
filename = 'UrbanSound8K/audio/fold1/9031-3-2-0.wav'
print_prediction(filename)
#playsound(filename)


#----- Validation/QC 2 - External audio-----
#New audio that weren't part of either test or training data

# External audio -  Dog
filename = 'UrbanSound8K/eval_audio/Evaluation_audio_dog_bark_1.wav'
print_prediction(filename)
#playsound(filename)

# External audio -  Jackhammer (wrong pred?)
filename = 'UrbanSound8K/eval_audio/Evaluation_audio_drilling_1.wav'
print_prediction(filename)
#playsound(filename)

# External audio -  Gun shot (wrong pred)
filename = 'UrbanSound8K/eval_audio/Evaluation_audio_gun_shot_1.wav'
print_prediction(filename)
#playsound(filename)

# External audio -  Siren
filename = 'UrbanSound8K/eval_audio/Evaluation_audio_siren_1.wav'
print_prediction(filename)
playsound(filename)

# #----- Evaluation Metrics-----

predicted_vector = np.argmax(model.predict(x_test), axis=-1)
y_pred = le.inverse_transform(predicted_vector) 
y_test_label = le.inverse_transform(np.argmax(y_test, axis=1))


import seaborn as sns
#Since I'm using Softmax the result is a prob vector
# Then, use np.argmax to get the most probable class
y_pred2 =  np.argmax(model.predict(x_test), axis=-1)
cf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred2)
#cf_matrix=confusion_matrix(y_test_label, y_pred)


#----- Model Performance with test Data----
print(classification_report(np.argmax(y_test, axis=1), y_pred2))

score_test = model.evaluate(x_test, y_test)
print("Testing Accuracy: ", score_test[1])



labels =['air_conditioner','car_horn','children_playing','dog_bark','drilling',
         'engine_idling','gun_shot','jackhammer','siren','street_music']



ax = sns.heatmap(cf_matrix, annot=True, fmt='g');
## Modify the Axes Object directly to set various attributes such as the
## Title, X/Y Labels.
ax.set_title('Clasification Report Urban sound');
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label');
## For the Tick Labels, the labels should be in Alphabetical order
#ax.xaxis.set_ticklabels(labels)
#ax.yaxis.set_ticklabels(labels)


#---- Confution Matrix Plot function ---
#https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Clasification Report Urban sound');
    # Note that due to returning the created figure object, when this funciton is called in a notebook
    # the figure willl be printed twice. To prevent this, either append ; to your function call, or
    # modify the function by commenting out this return expression.
    #return fig

#-- Confusion matrix figure 
print_confusion_matrix(cf_matrix, labels, figsize = (10,7), fontsize=14)


#====== Excecution time ======
print("--- %s seconds -> Total runing time---" % (time.time() - start_time))
















