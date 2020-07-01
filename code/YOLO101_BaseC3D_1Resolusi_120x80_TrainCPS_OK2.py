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
import c3d_model
from keras import backend as K
dim_ordering = K.image_dim_ordering()
backend = dim_ordering
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout3D
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score,roc_auc_score
#from keras.regularizers import l2, l1, WeightRegularizer
from keras.layers.normalization import BatchNormalization
import gc
from keras.models import model_from_json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from keras import layers, models, optimizers
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score,roc_auc_score,accuracy_score


def diagnose(data, verbose=True, label='input', plots=False, backend='tf'):
    # Convolution3D?
    if data.ndim > 2:
        if backend == 'th':
            data = np.transpose(data, (1, 2, 3, 0))
        #else:
        #    data = np.transpose(data, (0, 2, 1, 3))
        min_num_spatial_axes = 10
        max_outputs_to_show = 3
        ndim = data.ndim
        print "[Info] {}.ndim={}".format(label, ndim)
        print "[Info] {}.shape={}".format(label, data.shape)
        for d in range(ndim):
            num_this_dim = data.shape[d]
            if num_this_dim >= min_num_spatial_axes: # check for spatial axes
                # just first, center, last indices
                range_this_dim = [0, num_this_dim/2, num_this_dim - 1]
            else:
                # sweep all indices for non-spatial axes
                range_this_dim = range(num_this_dim)
            for i in range_this_dim:
                new_dim = tuple([d] + range(d) + range(d + 1, ndim))
                sliced = np.transpose(data, new_dim)[i, ...]
                print("[Info] {}, dim:{} {}-th slice: "
                      "(min, max, mean, std)=({}, {}, {}, {})".format(
                              label,
                              d, i,
                              np.min(sliced),
                              np.max(sliced),
                              np.mean(sliced),
                              np.std(sliced)))
        if plots:
            # assume (l, h, w, c)-shaped input
            if data.ndim != 4:
                print("[Error] data (shape={}) is not 4-dim. Check data".format(
                        data.shape))
                return
            l, h, w, c = data.shape
            if l >= min_num_spatial_axes or \
                h < min_num_spatial_axes or \
                w < min_num_spatial_axes:
                print("[Error] data (shape={}) does not look like in (l,h,w,c) "
                      "format. Do reshape/transpose.".format(data.shape))
                return
            nrows = int(np.ceil(np.sqrt(data.shape[0])))
            # BGR
            if c == 3:
                for i in range(l):
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                    im = np.squeeze(data[i, ...]).astype(np.float32)
                    im = im[:, :, ::-1] # BGR to RGB
                    # force it to range [0,1]
                    im_min, im_max = im.min(), im.max()
                    if im_max > im_min:
                        im_std = (im - im_min) / (im_max - im_min)
                    else:
                        print "[Warning] image is constant!"
                        im_std = np.zeros_like(im)
                    plt.imshow(im_std)
                    plt.axis('off')
                    plt.title("{}: t={}".format(label, i))
                plt.show()
                #plt.waitforbuttonpress()
            else:
                for j in range(min(c, max_outputs_to_show)):
                    for i in range(l):
                        mng = plt.get_current_fig_manager()
                        mng.resize(*mng.window.maxsize())
                        plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                        im = np.squeeze(data[i, ...]).astype(np.float32)
                        im = im[:, :, j]
                        # force it to range [0,1]
                        im_min, im_max = im.min(), im.max()
                        if im_max > im_min:
                            im_std = (im - im_min) / (im_max - im_min)
                        else:
                            print "[Warning] image is constant!"
                            im_std = np.zeros_like(im)
                        plt.imshow(im_std)
                        plt.axis('off')
                        plt.title("{}: o={}, t={}".format(label, j, i))
                    plt.show()
                    #plt.waitforbuttonpress()
    elif data.ndim == 1:
        print("[Info] {} (min, max, mean, std)=({}, {}, {}, {})".format(
                      label,
                      np.min(data),
                      np.max(data),
                      np.mean(data),
                      np.std(data)))
        print("[Info] data[:10]={}".format(data[:10]))

    return

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

R1x = 168
R1y = 112
R2x = 2
R2y = 3
R3x = 2
R3y = 3
RDepth = 16

kcv = 1
vartuning = '1Resolusi_168x112_C3DBase'
filenya = 'YOLO_U_Base_C3D_' + vartuning + '.csv'
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
    for labelIdx in range(1, 102):
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
                
                if not((count)%1 == 0):
                    continue
                frameR1 = cv2.resize(frame, (R1x, R1y), interpolation=cv2.INTER_AREA)
                frameR1 = frameR1[:, 28:140, :]
                framesR1.append(frameR1)
                frameR2 = cv2.resize(frame, (R2x, R2y), interpolation=cv2.INTER_AREA)
                framesR2.append(frameR2) 
                frameR3 = cv2.resize(frame, (R3x, R3y), interpolation=cv2.INTER_AREA)
                framesR3.append(frameR3)  
                
                #plt.imshow(gray, cmap = plt.get_cmap('gray'))
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                #plt.show()
                #cv2.imshow('frame',gray)
                if count == 16:
                    kondisi = False
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if not(count == 16):
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
                if not((count)%1 == 0):
                    continue
                frameR1 = cv2.resize(frame, (R1x, R1y), interpolation=cv2.INTER_AREA)
                frameR1 = frameR1[:, 28:140, :]
                framesR1.append(frameR1)
                frameR2 = cv2.resize(frame, (R2x, R2y), interpolation=cv2.INTER_AREA)
                framesR2.append(frameR2) 
                frameR3 = cv2.resize(frame, (R3x, R3y), interpolation=cv2.INTER_AREA)
                framesR3.append(frameR3)  

                #plt.imshow(gray, cmap = plt.get_cmap('gray'))
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                #plt.show()
                #cv2.imshow('frame',gray)
                if count == 16:
                    kondisi = False
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if not(count == 16):
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

    mean_cube = np.load('models/train01_16_128_171_mean.npy')
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
        
    train_set_R1 = np.zeros((count_train, RDepth, R1y,112,3))
    train_set_R2 = np.zeros((count_train, RDepth, R2y,R2x,3))
    train_set_R3 = np.zeros((count_train, RDepth, R3y,R3x,3))

    for h in xrange(count_train):
        train_set_R1[h][:][:][:][:]=X_train_R1_array[h]  - mean_cube[:, 8:120, 30:142, :]
        train_set_R2[h][:][:][:][:]=X_train_R2_array[h]
        train_set_R3[h][:][:][:][:]=X_train_R3_array[h]
        

    del X_train_R1_array
    del X_train_R2_array
    del X_train_R3_array
    gc.collect()
    
    test_set_R1 = np.zeros((count_test, RDepth, R1y,112,3))
    test_set_R2 = np.zeros((count_test, RDepth, R2y,R2x,3))
    test_set_R3 = np.zeros((count_test, RDepth, R3y,R3x,3))

    for h in xrange(count_test):
        test_set_R1[h][:][:][:][:]=X_test_R1_array[h]  - mean_cube[:, 8:120, 30:142, :]
        test_set_R2[h][:][:][:][:]=X_test_R2_array[h]
        test_set_R3[h][:][:][:][:]=X_test_R3_array[h]

    del X_test_R1_array 
    del X_test_R2_array 
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
        jumEpoch = 15
    nb_classes = 101
    
    model_dir = './models'
    global backend
    input_shape = (16, 112, 112, 3)
    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')
    model_c3d = model_from_json(open(model_json_filename, 'r').read())
    
    model_c3d.load_weights(model_weight_filename)
    
    print "sebelum popping layers c3d ==============="
    model_c3d.summary()
    
    model_c3d.pop()
    model_c3d.pop()
    model_c3d.pop()
    model_c3d.pop()
    model_c3d.pop()

    print "setelah popping layers c3d ==============="
    model_c3d.summary()        
    # buat menjadi functiuonal
    model_c3d.trainable = False
    model_c3d.build()
    model_c3d.trainable = False
    func_c3d = model_c3d.model
    func_c3d.trainable = False
    
    x = layers.Input(shape=input_shape)
    
    func_c3d = model_c3d.model (x)
    func_c3d.trainable = False
    
    model = BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None) (func_c3d)
    
    modelOut = Dense(nb_classes,init='normal', activation='softmax')(model)
    
    model = models.Model(inputs=x, outputs=modelOut)        

    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['acc'])
    
    
    # Train the model
    nama_filenya = "weights_" + vartuning +"_.hdf5" 

    checkpointer = ModelCheckpoint(filepath=nama_filenya, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
    hist = model.fit(train_set_R1, Y_train, validation_data=(test_set_R1, Y_test),
          batch_size=16, nb_epoch = jumEpoch, shuffle=True, verbose = 1, callbacks = [checkpointer])
          
     # Evaluate the model
    # load best model
    model.load_weights(nama_filenya)

    Y_pred = model.predict(test_set_R1, batch_size = 8)

    #print(Y_pred)
    k_val = 1
    Y_pred_label = []
    for idt in range(len(Y_pred)):
        Y_pred_label.append(np.argmax(Y_pred[idt]))
    print Y_test.shape    
    print Y_pred.shape
    print np.array(Y_pred_label).shape
    print np.argmax(Y_test,axis=1)
    print("Skor Model:")
    accScore = accuracy_score(np.argmax(Y_test,axis=1), Y_pred_label) 
    print(accScore)
    cohennya = cohen_kappa_score(np.argmax(Y_test,axis=1), Y_pred_label)
    print("kohen kappa:")
    print(cohennya)
    grup.append(k_val)
    grup.append(accScore)
    grup.append(cohennya)
    writer.writerows([grup])

    grup = []
