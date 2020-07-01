# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:28:09 2018

@author: aldi242
"""

import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import np_utils, generic_utils
import gc
from keras.layers.merge import concatenate
#from utils import combine_images
#import matplotlib.pyplot as plt
#from PIL import Image
import tensorflow as tf
from keras import initializers
#from keras.datasets import mnist
#from utils import plot_log
import cv2
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score,roc_auc_score,accuracy_score

#%% Classes

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



def load_videos(k_val, Rx1=12, Ry1=20, Rx2=12, Ry2=20, RzFaktor=8):
    # the data, shuffled and split between train and test sets
    testing = False
    faktor = RzFaktor

    R1x = Rx1
    R1y = Ry1
    R2x = Rx2
    R2y = Ry2
    R3x = 3
    R3y = 5
    RDepth = 52/faktor
    
    kcv = k_val

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
                
                if not((count)%faktor == 0):
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
                if not((count)%faktor == 0):
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
    test_set_R1 = np.zeros((count_test, RDepth, R1y,R1x,3))
    test_set_R2 = np.zeros((count_test, RDepth, R2y,R2x,3))
    test_set_R3 = np.zeros((count_test, RDepth, R3y,R3x,3))

    for h in xrange(count_test):
        test_set_R1[h][:][:][:][:]=X_test_R1_array[h]
        test_set_R2[h][:][:][:][:]=X_test_R2_array[h]
        test_set_R3[h][:][:][:][:]=X_test_R3_array[h]
        
    train_set_R1 = np.zeros((count_train, RDepth, R1y,R1x,3))
    train_set_R2 = np.zeros((count_train, RDepth, R2y,R2x,3))
    train_set_R3 = np.zeros((count_train, RDepth, R3y,R3x,3))

    for h in xrange(count_train):
        train_set_R1[h][:][:][:][:]=X_train_R1_array[h]
        train_set_R2[h][:][:][:][:]=X_train_R2_array[h]
        train_set_R3[h][:][:][:][:]=X_train_R3_array[h]
        
    del X_test_R1_array 
    del X_test_R2_array 
    del X_test_R3_array 
    gc.collect()
    del X_train_R1_array
    del X_train_R2_array
    del X_train_R3_array
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

    return (train_set_R1, train_set_R2, Y_train), (test_set_R1, test_set_R2, Y_test)


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param num_routing: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        """
        # Begin: routing algorithm V1, dynamic ------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        b = K.zeros(shape=[self.batch_size, self.num_capsule, self.input_num_capsule])

        def body(i, b, outputs):
            c = tf.nn.softmax(b, dim=1)  # dim=2 is the num_capsule dimension
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))
            if i != 1:
                b = b + K.batch_dot(outputs, inputs_hat, [2, 3])
            return [i-1, b, outputs]

        cond = lambda i, b, inputs_hat: i > 0
        loop_vars = [K.constant(self.num_routing), b, K.sum(inputs_hat, 2, keepdims=False)]
        shape_invariants = [tf.TensorShape([]),
                            tf.TensorShape([None, self.num_capsule, self.input_num_capsule]),
                            tf.TensorShape([None, self.num_capsule, self.dim_capsule])]
        _, _, outputs = tf.while_loop(cond, body, loop_vars, shape_invariants)
        # End: routing algorithm V1, dynamic ------------------------------------------------------------#
        """
        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # In forward pass, `inputs_hat_stopped` = `inputs_hat`;
        # In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
        inputs_hat_stopped = K.stop_gradient(inputs_hat)
        
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.num_routing - 1:
                # c.shape =  [batch_size, num_capsule, input_num_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
                # outputs.shape=[None, num_capsule, dim_capsule]
                outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]
            else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
                outputs = squash(K.batch_dot(c, inputs_hat_stopped, [2, 2]))

                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat_stopped, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding, nameK):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    outputC = layers.Conv3D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d_'+nameK)(inputs)
    conv1Caps = layers.MaxPooling3D(pool_size=(1, 1, 1), strides=None, padding='valid', data_format=None)(outputC)
    conv1Norm = layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv1Caps)    
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape_'+nameK)(conv1Norm)
    outputsDO = layers.Dropout(0.5)(outputs)
    return layers.Lambda(squash, name='primarycap_squash_'+nameK)(outputsDO)


"""
# The following is another way to implement primary capsule layer. This is much slower.
# Apply Conv2D `n_channels` times and concatenate all capsules
def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    outputs = []
    for _ in range(n_channels):
        output = layers.Conv2D(filters=dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        outputs.append(layers.Reshape([output.get_shape().as_list()[1] ** 2, dim_capsule])(output))
    outputs = layers.Concatenate(axis=1)(outputs)
    return layers.Lambda(squash)(outputs)
"""
def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


#%%


# --------------------------------------------- input datanya mnist
import csv
vartuning = '2Resolusi_240x180_160x120'
filenya = 'YOLO_U_VCPS_' + vartuning + '.csv'

Rx1 = 160
Ry1 = 120
Rx2 = 240
Ry2 = 180
RzFaktor = 4
with open(filenya, 'w') as out_file:
    writer = csv.writer(out_file, lineterminator = '\n')
    grup = []     
    grup.append('Blok ke-')
    grup.append('Skor Akurasi')
    grup.append('Skor Kappa')
    writer.writerows([grup])
    grup = []   

    for k_val in range(1,2):


        (x_train1, x_train2, y_train), (x_test1, x_test2, y_test) = load_videos(k_val, Rx1, Ry1, Rx2, Ry2, RzFaktor)

        scenario_name = str(Rx1) + 'x' + str(Ry1) + '_' + str(Rx2) + 'x' + str(Ry2) + '_' + str(RzFaktor)

        input_shape1 = (52/RzFaktor, Ry1, Rx1, 3)
        input_shape2 = (52/RzFaktor, Ry2, Rx2, 3)
        #print(x_train.shape[1:])
        n_class = len(np.unique(np.argmax(y_train, 1)))
        # print(np.argmax(y_train, 1))
        num_routing = 3
        batch_size = 16
        shift_fraction = 0.1
        lr = 0.001
        lam_recon = 0.392
        epochs = 45
        #  -------------------------------------------------- Definisikan Capsnet

        """
        A Capsule Network on MNIST.
        :param input_shape: data shape, 3d, [width, height, channels]
        :param n_class: number of classes
        :param num_routing: number of routing iterations
        :return: Two Keras Models, the first one used for training, and the second one for evaluation.
                `eval_model` can also be used for training.
        """
        x1 = layers.Input(shape=input_shape1)

        # Layer 1: Just a conventional Conv2D layer
        conv1t_1 = layers.Conv3D(filters=32, kernel_size=(3,7,7), strides=1, padding='valid', activation='relu', name='conv1t_1')(x1)
        conv1tMP_1 = layers.MaxPooling3D(pool_size=(1, 4, 4), strides=None, padding='valid', data_format=None)(conv1t_1)
        conv1tNorm_1 = layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv1tMP_1)
        conv1tDO_1 = layers.SpatialDropout3D(0.3)(conv1tNorm_1)       
        conv1_1 = layers.Conv3D(filters=32, kernel_size=(3,5,5), strides=1, padding='valid', activation='relu', name='conv1_1')(conv1tDO_1)
        conv1MP_1 = layers.MaxPooling3D(pool_size=(1, 4, 2), strides=None, padding='valid', data_format=None)(conv1_1)
        conv1Norm_1 = layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv1MP_1)
        conv1DO_1 = layers.SpatialDropout3D(0.5)(conv1Norm_1)
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps_1 = PrimaryCap(conv1DO_1, dim_capsule=8, n_channels=16, kernel_size=(3,3,3), strides=2, padding='valid', nameK = "1")

        x2 = layers.Input(shape=input_shape2)

        # Layer 1: Just a conventional Conv2D layer
        conv1t_2 = layers.Conv3D(filters=16, kernel_size=(3,7,7), strides=1, padding='valid', activation='relu', name='conv1t_2')(x2)
        conv1tMP_2 = layers.MaxPooling3D(pool_size=(1, 4, 6), strides=None, padding='valid', data_format=None)(conv1t_2)
        conv1tNorm_2 = layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv1tMP_2)
        conv1tDO_2 = layers.SpatialDropout3D(0.3)(conv1tNorm_2)       
        conv1_2 = layers.Conv3D(filters=32, kernel_size=(3,7,7), strides=1, padding='valid', activation='relu', name='conv1_2')(conv1tDO_2)
        conv1MP_2 = layers.MaxPooling3D(pool_size=(1, 6, 3), strides=None, padding='valid', data_format=None)(conv1_2)
        conv1Norm_2 = layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(conv1MP_2)
        conv1DO_2 = layers.SpatialDropout3D(0.5)(conv1Norm_2)
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps_2 = PrimaryCap(conv1DO_2, dim_capsule=8, n_channels=16, kernel_size=(3,3,3), strides=2, padding='valid', nameK = "2")


        # kombinasi kedua primary caps layer
        
        primarycaps_Kombi = concatenate([primarycaps_1, primarycaps_2],axis=1)

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=4, num_routing=num_routing,
                                 name='digitcaps')(primarycaps_Kombi)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        out_caps = Length(name='c_net')(digitcaps)

        # Decoder network.
        y = layers.Input(shape=(n_class,))
        masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

        print(np.prod(input_shape2))
        # Shared Decoder model in training and prediction
        decoder = models.Sequential(name='decoder')
        decoder.add(layers.Dense(128, activation='relu', input_dim=4*n_class))
        decoder.add(layers.Dense(np.prod(input_shape2)/1728, activation='relu'))
        decoder.add(layers.Reshape(target_shape=(52/(RzFaktor),Ry2/36,Rx2/48,3), name='out_recon_unupsized1'))
        decoder.add(layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
        decoder.add(layers.UpSampling3D(size=(1, 3, 4)))        
        decoder.add(layers.Conv3D(filters=16, kernel_size=(3,7,7), strides=1, padding='same', activation='relu', name='conv3Dout0'))
        decoder.add(layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
        decoder.add(layers.UpSampling3D(size=(1, 2, 3)))        
        decoder.add(layers.Conv3D(filters=16, kernel_size=(3,5,5), strides=1, padding='same', activation='relu', name='conv3Dout1'))
        decoder.add(layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))        
        decoder.add(layers.UpSampling3D(size=(1, 3, 2)))
        decoder.add(layers.Conv3D(filters=8, kernel_size=(3,5,5), strides=1, padding='same', activation='relu', name='conv3Dout2'))
        decoder.add(layers.BatchNormalization(epsilon=0.001, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
        decoder.add(layers.UpSampling3D(size=(1, 2, 2)))
        decoder.add(layers.Conv3D(filters=3, kernel_size=(3,3,3), strides=1, padding='same', activation='sigmoid', name='conv3Dout3'))


        # Models for training and evaluation (prediction)
        model = models.Model([x1, x2, y], [out_caps, decoder(masked_by_y)])
        eval_model = models.Model([x1, x2], [out_caps, decoder(masked)])

        # manipulate model
        noise = layers.Input(shape=(n_class, 4))
        noised_digitcaps = layers.Add()([digitcaps, noise])
        masked_noised_y = Mask()([noised_digitcaps, y])
        manipulate_model = models.Model([x1, x2, y, noise], decoder(masked_noised_y))



        model.summary()


        namafilenya = './result/' + '_' + scenario_name  + '_validasiK_' + str(k_val) + '_' + 'weights.h5'
        tb = callbacks.TensorBoard(log_dir='./sken4', batch_size=2, histogram_freq=0, write_graph=True,
                                   write_images=True)
        log = callbacks.CSVLogger('./result' + '/log' + '_' + scenario_name + '_validasiK_' + str(1) + '_' + '.csv')
        checkpoint = callbacks.ModelCheckpoint(namafilenya, monitor='val_c_net_acc',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))

        # compile the model
        model.compile(optimizer=optimizers.Adam(lr=lr),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., lam_recon],
                      metrics={'c_net': 'accuracy'})

        """
        # Training without data augmentation:
        model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
                  validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
        """

        # Begin: Training with data augmentation ---------------------------------------------------------------------#
        def train_generator(x, y, batch_size, shift_fraction=0.1):
            train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                               height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
            generator = train_datagen.flow(x, y, batch_size=batch_size)
            while 1:
                x_batch, y_batch = generator.next()
                yield ([x_batch, y_batch], [y_batch, x_batch])

        # Training with data augmentation. If shift_fraction=0., also no augmentation.
        model.fit(x = [x_train1, x_train2, y_train], y = [y_train, x_train2],
                            epochs=epochs, shuffle=True, batch_size = batch_size,
                            validation_data=[[x_test1, x_test2, y_test], [y_test, x_test2]],
                            callbacks=[log, tb, checkpoint, lr_decay])
        # End: Training with data augmentation -----------------------------------------------------------------------#
        #model.save_weights('./result/'  + '_' + scenario_name + '_validasiK_' + str(k_val) + '_' + 'trained_model.h5')
        #print('Trained model saved to \'%s/trained_model.h5\'' % './result')


        #plot_log('./result' + '/log.csv', show=True)
        model.load_weights(namafilenya)

        Y_pred = model.predict([x_test1, x_test2, y_test], batch_size = 8)

        #print(Y_pred)

        Y_pred_label = []
        for idt in range(len(Y_pred[0])):
            Y_pred_label.append(np.argmax(Y_pred[0][idt]))
        print("Skor Model:")
        accScore = accuracy_score(np.argmax(y_test,axis=1), Y_pred_label) 
        print(accScore)
        cohennya = cohen_kappa_score(np.argmax(y_test,axis=1), Y_pred_label)
        print("kohen kappa:")
        print(cohennya)
        grup.append(k_val)
        grup.append(accScore)
        grup.append(cohennya)
        writer.writerows([grup])

        grup = []




