# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:24:34 2017

@author: user
"""

#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout3D, Merge
#from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils, generic_utils
import os
import pandas as pd
import matplotlib
from keras.callbacks import ModelCheckpoint
import keras.callbacks
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout3D
from keras.engine.topology import Input
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score,roc_auc_score
#from keras.regularizers import l2, l1, WeightRegularizer
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
import gc
from keras.models import Model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def getLabelFromIdx(x):
    return {
        1 : 'ApplyEyeMakeup',
        2 : 'ApplyLipstick',
        3 : 'Archery',
        4 : 'BabyCrawling',
        5 : 'BalanceBeam',
        6 : 'BandMarching',
        7 : 'BaseballPitch',
        8 : 'Basketball',
        9 : 'BasketballDunk',
        10 : 'BenchPress',
        11 : 'Biking',
        12 : 'Billiards', 
        13 : 'BlowDryHair',
        14 : 'BlowingCandles',
        15 : 'BodyWeightSquats', 
        16 : 'Bowling',
        17 : 'BoxingPunchingBag',
        18 : 'BoxingSpeedBag',
        19 : 'BreastStroke',
        20 : 'BrushingTeeth',
        21 : 'CleanAndJerk',
        22 : 'CliffDiving',
        23 : 'CricketBowling',
        24 : 'CricketShot',
        25 : 'CuttingInKitchen',
        26 : 'Diving',
        27 : 'Drumming',
        28 : 'Fencing', 
        29 : 'FieldHockeyPenalty',
        30 : 'FloorGymnastics',
        31 : 'FrisbeeCatch',
        32 : 'FrontCrawl',
        33 : 'GolfSwing',
        34 : 'Haircut', 
        35 : 'Hammering',
        36 : 'HammerThrow',
        37 : 'HandstandPushups',
        38 : 'HandstandWalking',
        39 : 'HeadMassage',
        40 : 'HighJump',
        41 : 'HorseRace',
        42 : 'HorseRiding',
        43 : 'HulaHoop',
        44 : 'IceDancing',
        45 : 'JavelinThrow',
        46 : 'JugglingBalls',
        47 : 'JumpingJack',
        48 : 'JumpRope',
        49 : 'Kayaking',
        50 : 'Knitting',
        51 : 'LongJump',
        52 : 'Lunges',
        53 : 'MilitaryParade',
        54 : 'Mixing',
        55 : 'MoppingFloor',
        56 : 'Nunchucks',
        57 : 'ParallelBars',
        58 : 'PizzaTossing',
        59 : 'PlayingCello',
        60 : 'PlayingDaf',
        61 : 'PlayingDhol',
        62 : 'PlayingFlute',
        63 : 'PlayingGuitar',
        64 : 'PlayingPiano',
        65 : 'PlayingSitar',
        66 : 'PlayingTabla',
        67 : 'PlayingViolin',
        68 : 'PoleVault',
        69 : 'PommelHorse',
        70 : 'PullUps',
        71 : 'Punch',
        72 : 'PushUps',
        73 : 'Rafting',
        74 : 'RockClimbingIndoor',
        75 : 'RopeClimbing',
        76 : 'Rowing',
        77 : 'SalsaSpin',
        78 : 'ShavingBeard',
        79 : 'Shotput',
        80 : 'SkateBoarding',
        81 : 'Skiing',
        82 : 'Skijet',
        83 : 'SkyDiving',
        84 : 'SoccerJuggling',
        85 : 'SoccerPenalty',
        86 : 'StillRings',
        87 : 'SumoWrestling',
        88 : 'Surfing',
        89 : 'Swing',
        90 : 'TableTennisShot',
        91 : 'TaiChi',
        92 : 'TennisSwing',
        93 : 'ThrowDiscus',
        94 : 'TrampolineJumping',
        95 : 'Typing',
        96 : 'UnevenBars',
        97 : 'VolleyballSpiking',
        98 : 'WalkingWithDog',
        99 : 'WallPushups',
        100 : 'WritingOnBoard',
        101 : 'YoYo'
    }.get(x, "----")


testing = False
R1x = 240
R1y = 180
R2x = 160
R2y = 120
R3x = 2
R3y = 3
RDepth = 13

kcv = 1
vartuning = '2Resolusi_160x120_240x180'
filenya = 'YOLO_U_V1_' + vartuning + '.csv'
with open(filenya, 'w') as out_file:
    writer = csv.writer(out_file, lineterminator = '\n')
    grup = []     
    grup.append('Blok ke-')
    grup.append('Skor Akurasi')
    grup.append('Skor Kappa')
    writer.writerows([grup])
    grup = []   
    X_train_R1 = []
    X_train_R2 = []
    X_train_R3 = []
    labels_train = []
    count_train = 0
    X_test_R1 = []
    X_test_R2 = []
    X_test_R3 = []
    labels_test = []
    count_test = 0

    # training data input
    for labelIdx in range(1, 101):
        print labelIdx
        listing = os.listdir('TestData/' + getLabelFromIdx(labelIdx) + '/')
        
        count_pretesting = 0
        
        for vid in listing: 
            count_pretesting += 1
#
            if (count_pretesting > 5) and testing:
                break                 
            vid = 'TestData/' + getLabelFromIdx(labelIdx) + '/' +vid
            framesR1 = []
            framesR2 = []
            framesR3 = []
            cap = cv2.VideoCapture(vid)
            fps = cap.get(5)
            #print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
            
            #test 
            frame = []
            ret, frame = cap.read()
            
            #print frame.shape
            if frame is None:
                print "image not readable"
                break
            
            
            
            count = 0
            kondisi = True
            while kondisi == True:
                ret, frame = cap.read()
                if frame is None:
                    print "skipping vid"
                    break
                count += 1
                
                if not((count)%4 == 0):
                    continue
                frameR1 = cv2.resize(frame, (R1x, R1y), interpolation=cv2.INTER_AREA)
                framesR1.append(frameR1)
                frameR2 = cv2.resize(frame, (R2x, R2y), interpolation=cv2.INTER_AREA)
                framesR2.append(frameR2) 
                frameR3 = cv2.resize(frame, (R3x, R3y), interpolation=cv2.INTER_AREA)
                framesR3.append(frameR3)  
                
                #plt.imshow(gray, cmap = plt.get_cmap('gray'))
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                #plt.show()
                #cv2.imshow('frame',gray)
                if count == 52:
                    kondisi = False
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if not(count == 52):
                print "vid not saved"
                continue
            count_test += 1
            label = labelIdx-1
            labels_test.append(label)
            cap.release()
            cv2.destroyAllWindows()
        
            inputR1=np.array(framesR1)
            inputR2=np.array(framesR2)
            inputR3=np.array(framesR3)
        
            #print input.shape
            iptR1=inputR1
            iptR2=inputR2
            iptR3=inputR3
            #print ipt.shape
        
            X_test_R1.append(iptR1)
            X_test_R2.append(iptR2)
            X_test_R3.append(iptR3)

        listing = os.listdir('TrainData/' + getLabelFromIdx(labelIdx) + '/')
        
        count_pretesting = 0
        for vid in listing:
            count_pretesting += 1

            if (count_pretesting > 5) and testing:
                break                      
            vid = 'TrainData/' + getLabelFromIdx(labelIdx) + '/' +vid
            framesR1 = []
            framesR2 = []
            framesR3 = []
            cap = cv2.VideoCapture(vid)
            fps = cap.get(5)
            #print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
            #test 
            frame = []
            ret, frame = cap.read()
            
            #print frame.shape
            if frame is None:
                print "image not readable"
                break
            
            
            
            count = 0
            kondisi = True
            while kondisi == True:
                ret, frame = cap.read()
                if frame is None:
                    print "skipping vid"
                    break
                count += 1
                if not((count)%4 == 0):
                    continue
                frameR1 = cv2.resize(frame, (R1x, R1y), interpolation=cv2.INTER_AREA)
                framesR1.append(frameR1)
                frameR2 = cv2.resize(frame, (R2x, R2y), interpolation=cv2.INTER_AREA)
                framesR2.append(frameR2) 
                frameR3 = cv2.resize(frame, (R3x, R3y), interpolation=cv2.INTER_AREA)
                framesR3.append(frameR3)  

                #plt.imshow(gray, cmap = plt.get_cmap('gray'))
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                #plt.show()
                #cv2.imshow('frame',gray)
                if count == 52:
                    kondisi = False
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if not(count == 52):
                print "vid not saved"
                continue
            count_train += 1
            label = labelIdx-1
            labels_train.append(label)
            cap.release()
            cv2.destroyAllWindows()
        
            inputR1=np.array(framesR1)
            inputR2=np.array(framesR2)
            inputR3=np.array(framesR3)
        
            #print input.shape
            iptR1=inputR1
            iptR2=inputR2
            iptR3=inputR3
            #print ipt.shape
        
            X_train_R1.append(iptR1)
            X_train_R2.append(iptR2)
            X_train_R3.append(iptR3)

    # formatting data                
    X_train_R1_array = (X_train_R1) 
    X_train_R2_array = (X_train_R2) 
    X_train_R3_array = (X_train_R3) 
    labels_train_array = np.array(labels_train)
    Y_train = np_utils.to_categorical(labels_train_array, 101)
    
    del X_train_R1 
    del X_train_R2 
    del X_train_R3 
    gc.collect()
    X_test_R1_array = (X_test_R1) 
    X_test_R2_array = (X_test_R2) 
    X_test_R3_array = (X_test_R3) 
    labels_test_array = np.array(labels_test)
    Y_test = np_utils.to_categorical(labels_test_array, 101)
    
    del X_test_R1
    del X_test_R2 
    del X_test_R3 
    gc.collect()
    
    train_set_R1 = np.zeros((count_train, RDepth, R1y,R1x,3))
    for h in xrange(count_train):
        train_set_R1[h][:][:][:][:]=X_train_R1_array[h]

    del X_train_R1_array
    gc.collect()
                
    train_set_R2 = np.zeros((count_train, RDepth, R2y,R2x,3))
    for h in xrange(count_train):
        train_set_R2[h][:][:][:][:]=X_train_R2_array[h]

    del X_train_R2_array
    gc.collect()
        
    train_set_R3 = np.zeros((count_train, RDepth, R3y,R3x,3))
    for h in xrange(count_train):
        train_set_R3[h][:][:][:][:]=X_train_R3_array[h]

    del X_train_R3_array
    gc.collect()        

    
    
    test_set_R1 = np.zeros((count_test, RDepth, R1y,R1x,3))    
    for h in xrange(count_test):
        test_set_R1[h][:][:][:][:]=X_test_R1_array[h]

    del X_test_R1_array 
    gc.collect()
        
    test_set_R2 = np.zeros((count_test, RDepth, R2y,R2x,3))
    for h in xrange(count_test):
        test_set_R2[h][:][:][:][:]=X_test_R2_array[h]

    del X_test_R2_array 
    gc.collect()        
        
    test_set_R3 = np.zeros((count_test, RDepth, R3y,R3x,3))
    for h in xrange(count_test):
        test_set_R3[h][:][:][:][:]=X_test_R3_array[h]

    del X_test_R3_array 
    gc.collect()
        

    train_set_R1 = train_set_R1.astype('float32')
    
    train_set_R1 -= 127.5
    
    train_set_R1 /=127.5

    train_set_R2 = train_set_R2.astype('float32')
    
    train_set_R2 -= 127.5
    
    train_set_R2 /=127.5
    train_set_R3 = train_set_R3.astype('float32')
    
    train_set_R3 -= 127.5
    
    train_set_R3 /=127.5
    

    test_set_R1 = test_set_R1.astype('float32')
    
    test_set_R1 -= 127.5
    
    test_set_R1 /=127.5

    test_set_R2 = test_set_R2.astype('float32')
    
    test_set_R2 -= 127.5
    
    test_set_R2 /=127.5
    test_set_R3 = test_set_R3.astype('float32')
    
    test_set_R3 -= 127.5
    
    test_set_R3 /=127.5
    
    #%% definisikan sebuah model
    
#        # Parameter tuning
#        jumEpoch = 25
#        nb_classes = 8
#        #Lengan A
#        filterNumL1 = 16 # jumlah filter L1
#        filterSizeXYL1 = 5  #ukuran filter dimensi spasial
#        filterSizeTL1 = 3#ukuran filter dimensi spasial
#        
#        poolingSizeXYL1 = 3
#        poolingSizeTL1 = 1
#        poolingStrideXYL1 = 1
#        poolingStrideTL1 = 1  #parameter pooling L1
#        #Lengan B
#        filterNumL1B = 32 # jumlah filter L1
#        filterSizeXYL1B = 3  #ukuran filter dimensi spasial
#        filterSizeTL1B = 3 #ukuran filter dimensi spasial
#        
#        poolingSizeXYL1B = 3
#        poolingSizeTL1B = 1
#        poolingStrideXYL1B = 1
#        poolingStrideTL1B = 1  #parameter pooling L1
    
    # Define model
    
#        modelA = Sequential()
#        modelA.add(Convolution3D(filterNumL1,kernel_dim1=filterSizeXYL1, kernel_dim2=filterSizeXYL1, kernel_dim3=filterSizeTL1, input_shape=(10, 20, 30, 3), activation='relu', dim_ordering='tf'))            
#        modelA.add(MaxPooling3D(pool_size=(poolingSizeXYL1, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf'))
#        modelA.add(SpatialDropout3D(0.4))         
#        modelA.add(Flatten())
#        
#        modelB = Sequential()
#        modelB.add(Convolution3D(filterNumL1B,kernel_dim1=filterSizeXYL1B, kernel_dim2=filterSizeXYL1B, kernel_dim3=filterSizeTL1B, input_shape=(10, 20, 30, 3), activation='relu', dim_ordering='tf'))            
#        modelB.add(MaxPooling3D(pool_size=(poolingSizeXYL1B, poolingSizeXYL1B, poolingSizeTL1B), dim_ordering='tf'))
#        modelB.add(SpatialDropout3D(0.4))         
#        modelB.add(Flatten())
#
#
#        model = Sequential() 
#        model.add(Merge([modelA, modelB], mode='concat'))
#        model.add(Dense(paramuji, init='normal', activation='relu'))
#        
#        model.add(Dropout(0.4))
#        
#        model.add(Dense(nb_classes,init='normal'))
#        
#        model.add(Activation('softmax'))
#        model.summary()
    
#        model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics = ["accuracy"])
#        
#        
#        # Train the model
#        
#        hist = model.fit([train_set, train_set], Y_train, validation_data=([test_set, test_set], Y_test),
#              batch_size=15, nb_epoch = jumEpoch, show_accuracy=True, shuffle=True, verbose = 0)
#              
#         # Evaluate the model
#        score = model.evaluate([test_set, test_set], Y_test, batch_size=15, show_accuracy=True)
#    
    # Define model
    # Parameter tuning
    if testing:
        jumEpoch = 2
    else:        
        jumEpoch = 250
    nb_classes = 101
    
    filterNumL1 = 64 # jumlah filter L1
    filterSizeXYL1 = 5  #ukuran filter dimensi spasial
    filterSizeTL1 = 3#ukuran filter dimensi spasial
    
    poolingSizeXYL1 = 2
    poolingSizeTL1 = 2
    poolingStrideXYL1 = 1
    poolingStrideTL1 = 1  #parameter pooling L1
    
    filterNumL2 = 64 # jumlah filter L1
    filterSizeXYL2 = 5  #ukuran filter dimensi spasial
    filterSizeTL2 = 5#ukuran filter dimensi spasial        

    modelB_In = Input(shape=(RDepth, R1y, R1x,3)) 
    modelB = Convolution3D(32,kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, input_shape=(RDepth, R1y, R1x,3), activation='relu', dim_ordering='tf')(modelB_In)
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelB = MaxPooling3D(pool_size=(1, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelB)
    modelB = SpatialDropout3D(0.3)(modelB)
    modelB = Convolution3D(filterNumL2,kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf')(modelB)
    # model.add(Convolution3D(filterNumL2,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf'))
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelB = MaxPooling3D(pool_size=(1, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelB)                    
    modelB = SpatialDropout3D(0.3)(modelB)
    modelB = Convolution3D(filterNumL2,kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf')(modelB)
    # model.add(Convolution3D(filterNumL2,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf'))
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelB = MaxPooling3D(pool_size=(2, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelB)                    
    modelB = SpatialDropout3D(0.3)(modelB)
    modelB = Convolution3D(128,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf')(modelB)
    # model.add(Convolution3D(128,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf'))
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelB = MaxPooling3D(pool_size=(1, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelB)                    
    modelB = SpatialDropout3D(0.5)(modelB)
    modelB = Convolution3D(128,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf')(modelB)
    # model.add(Convolution3D(128,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf'))
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelB = MaxPooling3D(pool_size=(1, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelB)                    
    modelB = SpatialDropout3D(0.5)(modelB)         
    modelB = Flatten()(modelB)  

    modelA_In = Input(shape=(RDepth, R2y, R2x,3))
    modelA = Convolution3D(filterNumL1,kernel_dim1=filterSizeTL1, kernel_dim2=filterSizeXYL1, kernel_dim3=filterSizeXYL1, input_shape=(RDepth, R1y, R1x,3), activation='relu', dim_ordering='tf')(modelA_In)
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelA = MaxPooling3D(pool_size=(1, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelA)
    modelA = SpatialDropout3D(0.3)(modelA)
    modelA = Convolution3D(filterNumL2,kernel_dim1=filterSizeTL2, kernel_dim2=filterSizeXYL2, kernel_dim3=filterSizeXYL2, activation='relu', dim_ordering='tf')(modelA)
    # model.add(Convolution3D(filterNumL2,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf'))
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelA = MaxPooling3D(pool_size=(2, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelA)                    
    modelA = SpatialDropout3D(0.3)(modelA) 
    modelA = Convolution3D(128,kernel_dim1=3, kernel_dim2=5, kernel_dim3=5, activation='relu', dim_ordering='tf')(modelA)
    # model.add(Convolution3D(128,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf'))
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelA = MaxPooling3D(pool_size=(1, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelA)                    
    modelA = SpatialDropout3D(0.5)(modelA)
    modelA = Convolution3D(128,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf')(modelA)
    # model.add(Convolution3D(128,kernel_dim1=1, kernel_dim2=3, kernel_dim3=3, activation='relu', dim_ordering='tf'))
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    modelA = MaxPooling3D(pool_size=(1, poolingSizeXYL1, poolingSizeTL1), dim_ordering='tf')(modelA)                    
    modelA = SpatialDropout3D(0.5)(modelA)         
    modelA = Flatten()(modelA)      
    
    modelG = concatenate([modelB, modelA])       
    modelG = BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(modelG)        
    modelG_Out = Dense(nb_classes, init='glorot_normal', activation='softmax')(modelG)
    model = Model(inputs=[modelB_In, modelA_In], outputs = modelG_Out)
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['acc'])
    
    
    # Train the model
    nama_filenya = "weights_" + vartuning +"_.hdf5" 
    
    checkpointer = ModelCheckpoint(filepath=nama_filenya, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
    hist = model.fit([train_set_R1, train_set_R2], Y_train, validation_data=([test_set_R1, test_set_R2], Y_test),
    batch_size=16, nb_epoch = jumEpoch, shuffle=True, verbose = 1, callbacks = [checkpointer])
      
    # Evaluate the model
    # load best model
      
    model.load_weights(nama_filenya)
    score = model.evaluate([test_set_R1, test_set_R2], Y_test, batch_size=8)
    print "Skor Model:"
    print score[1]
    Y_proba = model.predict([test_set_R1, test_set_R2], batch_size = 8)
    print Y_proba
    Y_pred = np.argmax(Y_proba, axis=1)
    print Y_pred
    grup.append(kcv)
    grup.append(score[1])
    cohennya = cohen_kappa_score(np.argmax(Y_test,axis=1), Y_pred)
    print "kohen kappa:"
    print cohennya
    grup.append(cohennya)
    writer.writerows([grup])
