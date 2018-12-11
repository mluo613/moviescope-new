from config.global_parameters import default_model_name, number_of_classes, number_of_frames
from utils import load_pkl, augment_labels_lstm, gather_features, gather_raw_data, dump_pkl
from video import sequencify
import numpy as np
from model_utils import text_model, vis_model, good_text_model
from keras.optimizers import SGD
import keras.backend as K
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Merge, Dense, Dropout, Embedding, Input, LSTM, merge, BatchNormalization, Flatten, Reshape,Lambda
from keras.utils.visualize_util import plot
from bilinear_tensor import BilinearTensorLayer

remote = callbacks.RemoteMonitor(root='http://128.143.63.199:9009')


def bilinear_projection(inputs):
    x, y = inputs
    batch_size = K.shape(x)[0]
    outer_product = x[:,:,np.newaxis] * y[:,np.newaxis,:]
    return K.reshape(outer_product,(batch_size, -1))


def train_classifier_video(trainVideoFeatures, trainLabels, valVideoFeatures=None, valLabels=None):

    input_dim = 4096

    trainingLabels, trainingFeatures = augment_labels_lstm(trainLabels, trainVideoFeatures, number_of_frames)

    print trainingLabels.shape
    print trainingFeatures.shape

    """Initialize the mode"""

    visInput = Input(shape=(number_of_frames, input_dim), dtype='float32')
    model = vis_model(visInput, number_of_classes, return_top=True)

    plot(model, to_file='vis_model.png', show_shapes=True)
    sgd = SGD(lr=0.01, decay=0.000001, momentum=0.9, nesterov=True)
# suppressing SGD, since text and merged models are optimized using ADAM

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    """Start training"""
    batch_size = 63
    nb_epoch = 50

    checkpoint = ModelCheckpoint(filepath='./data/models/wiki_im_video_sgd.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, remote]

    if valLabels is not None:
        valLabels, valFeatures = augment_labels_lstm(valLabels, valVideoFeatures, number_of_frames)
        hist = model.fit(trainingFeatures, trainingLabels, validation_data=(valFeatures, valLabels), nb_epoch=nb_epoch,batch_size=batch_size, callbacks=callbacks_list)
    else:
        hist = model.fit(trainingFeatures, trainingLabels, nb_epoch=nb_epoch,batch_size=batch_size, callbacks=callbacks_list)
    model.save('data/models/video_sgd.h5')
    histDict = hist.history
    dump_pkl(histDict, 'hist_video_sgd')

    return model 
 

def train_classifier_word_embedding(trainPlotFeatures, trainLabels, valPlotFeatures=None, valLabels=None):

    sequence_input = Input(shape=(3000,), dtype='int32')
    sequence_input_reverse = Input(shape=(3000,), dtype='int32')

    textModel = good_text_model(sequence_input, sequence_input_reverse, use_embedding=True, trainable=False)
    textModel.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='./data/models/text_checkpoint.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint_loss = ModelCheckpoint(filepath='./data/models/text_checkpoint_loss.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, remote, checkpoint_loss]

    if valLabels is not None:
        hist = textModel.fit([trainPlotFeatures[0], trainPlotFeatures[1]], trainLabels, validation_data=([valPlotFeatures[0], valPlotFeatures[1]], valLabels),nb_epoch=50,batch_size=63, callbacks=callbacks_list)
    else:
        hist = textModel.fit([trainPlotFeatures[0], trainPlotFeatures[1]], trainLabels, nb_epoch=100,batch_size=63, callbacks=callbacks_list)

    textModel.save('data/models/_text.h5')
    print "Model saved at: data/models/_text.h5"
    histDict = hist.history
    dump_pkl(histDict, 'hist_text')


def train_classifier_vislang(trainVideoFeatures, trainPlotFeatures, trainLabels, valVideoFeatures=None,
valPlotFeatures=None, valLabels=None, merge_mode='concat'):

    input_dim = 4096

    ## Text modelling

    sequence_input = Input(shape=(3000,), dtype='int32')
    sequence_input_reverse = Input(shape=(3000,), dtype='int32')
    textModel = good_text_model(sequence_input, sequence_input_reverse, return_top=False)

    ## video modelling

    trainingLabels, trainVideoFeatures = augment_labels_lstm(trainLabels, trainVideoFeatures, number_of_frames)

    print trainVideoFeatures.shape
    print trainingLabels.shape

    if valLabels is not None:
        valLabels, valVideoFeatures = augment_labels_lstm(valLabels, valVideoFeatures, number_of_frames)

    visInput = Input(shape=(number_of_frames, input_dim), dtype='float32')
    visModel = vis_model(visInput, number_of_classes, return_top=False)

    if False:
        # save precomputed features
        _model = Model(input=visInput, output=visModel)
        visFeatures = _model.predict(trainVideoFeatures)
        dump_pkl(visFeatures, 'train_visFeatures')
        _model = Model(input=[sequence_input, sequence_input_reverse], output=[textModel])
        textFeatures = _model.predict([trainPlotFeatures[0], trainPlotFeatures[1]])
        dump_pkl(textFeatures, 'train_textFeatures')

    if merge_mode == 'bilinear':
        predictionLayer = BilinearTensorLayer(input_dim=64)([visModel, textModel])
    else:
        if merge_mode in ['concat','sum','mul']:
            vislangModel = merge([visModel, textModel], mode=merge_mode, name='vislang')
        else:
            vislangModel = Lambda(bilinear_projection, output_shape=(4096,))([visModel, textModel])
            vislangModel = Dense(1024, activation='relu')(vislangModel)

        if merge_mode == 'sum':
            vislangModel = Dense(1000, activation='relu')(vislangModel)
            vislangModel = Dropout(0.5)(vislangModel)
            vislangModel = Dense(8, activation='relu')(vislangModel)
        elif merge_mode == 'concat':
            vislangModel = Dense(512, activation='relu')(vislangModel)
            vislangModel = Dropout(0.5)(vislangModel)
            vislangModel = Dense(14, activation='relu')(vislangModel)
        elif  merge_mode =='outer':
            vislangModel = Dense(16, activation='relu')(vislangModel)
            vislangModel = Dropout(0.5)(vislangModel)
            vislangModel = Dense(256, activation='relu')(vislangModel)

       
#        vislangModel = Dense(64, activation='relu')(vislangModel)
        vislangModel = Dropout(0.25)(vislangModel)
        predictionLayer = Dense(number_of_classes, activation='sigmoid', name='main_output')(vislangModel)

    model = Model(input=[visInput, sequence_input, sequence_input_reverse], output=[predictionLayer])
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    ##
    if True:
        model.load_weights('data/weights/eq_weights_min_loss_%s.h5' % merge_mode)
    ##

    plot(model, to_file='vislang_model_%s.png' % merge_mode, show_shapes=True)

    checkpoint = ModelCheckpoint(filepath='./data/models/wiki_im_eq_vislang_%s.h5' % merge_mode, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint_loss = ModelCheckpoint(filepath='./data/weights/eq_weights_min_loss_%s.h5' % merge_mode, monitor='val_loss', save_weights_only=True, mode='min')
    callbacks_list = [checkpoint, remote, checkpoint_loss]

    if valLabels is not None:
        hist = model.fit(x=[trainVideoFeatures, trainPlotFeatures[0], trainPlotFeatures[1]], y=trainLabels,
        validation_data=([valVideoFeatures, valPlotFeatures[0], valPlotFeatures[1]], valLabels),nb_epoch=30,batch_size=63, callbacks=callbacks_list)
    else:
        hist = model.fit(x=[trainVideoFeatures, trainPlotFeatures[0], trainPlotFeatures[1]], y=trainLabels, nb_epoch=1,batch_size=63, callbacks=callbacks_list)

    model.save('data/models/eq_VisLang_%s.h5' % merge_mode)
    model.save_weights('eq_%s_weights.h5' % merge_mode)
    histDict = hist.history
    dump_pkl(histDict, 'hist_eq_vislang_%s' % merge_mode)

    return model


def fine_tune_merge_only(merge_mode='sum'):

    if False:
        trainLabels = gather_features(mode='train', return_plot=False, return_video=False)
        valLabels = gather_features(mode='val', return_plot=False, return_video=False)

        dump_pkl(trainLabels, 'trainLabels')
        dump_pkl(valLabels, 'valLabels')
    else:
        trainLabels = load_pkl('trainLabels')
        valLabels = load_pkl('valLabels')

    train_visFeatures = load_pkl('train_visFeatures')
    train_textFeatures = load_pkl('train_textFeatures')

    val_visFeatures = load_pkl('val_visFeatures')
    val_textFeatures = load_pkl('val_textFeatures')

    visInput = Input(shape=(64,))
    textInput = Input(shape=(64,))

    if merge_mode in ['sum', 'concat', 'mul', 'outer']:
        if merge_mode == 'outer':
            vislangModel = Lambda(bilinear_projection, output_shape=(4096,))([visInput, textInput])
        else:
            vislangModel = merge([visInput, textInput], mode=merge_mode)

        if merge_mode == 'sum':
            vislangModel = Dense(1000, activation='relu')(vislangModel)
            vislangModel = Dropout(0.5)(vislangModel)
            vislangModel = Dense(8, activation='relu')(vislangModel)
        elif merge_mode == 'concat':
            vislangModel = Dense(512, activation='relu')(vislangModel)
            vislangModel = Dropout(0.5)(vislangModel)
            vislangModel = Dense(14, activation='relu')(vislangModel)
        elif  merge_mode =='outer':
            vislangModel = Dense(16, activation='relu')(vislangModel)
            vislangModel = Dropout(0.5)(vislangModel)
            vislangModel = Dense(256, activation='relu')(vislangModel)

        vislangModel = Dense(number_of_classes, activation='sigmoid')(vislangModel)
    else:
        vislangModel = BilinearTensorLayer(input_dim=64)([visInput, textInput])
    
    sgd = SGD(lr=0.1, decay=0.00001, momentum=0.9, nesterov=True)
    model = Model(input=[visInput, textInput], output=[vislangModel])
    model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath='./data/models/ft_vislang_%s.h5' % merge_mode, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint_loss = ModelCheckpoint(filepath='./data/weights/ft_weights_min_loss_%s.h5' % merge_mode, monitor='val_loss', save_weights_only=True, mode='min')
    callbacks_list = [checkpoint, remote, checkpoint_loss]

    model.load_weights('data/weights/ft_weights_min_loss_%s.h5' % merge_mode)
    hist = model.fit(x=[train_visFeatures, train_textFeatures], y=trainLabels, validation_data=([val_visFeatures,
    val_textFeatures], valLabels), nb_epoch=50, batch_size=128, callbacks=callbacks_list)

    histDict = hist.history
    dump_pkl(histDict, 'hist_ft_vislang_%s' % merge_mode)

def main_vislang():
    trainLabels, trainPlotFeatures, trainVideoFeatures = gather_features('train', reverse=True)
    valLabels, valPlotFeatures, valVideoFeatures = gather_features('val', reverse=True)
    trainPlotFeatures = np.array(map(list, zip(*trainPlotFeatures)))
    valPlotFeatures = np.array(map(list, zip(*valPlotFeatures)))
#    train_classifier_vislang(valVideoFeatures, valPlotFeatures, valLabels, merge_mode='outer')
    train_classifier_vislang(trainVideoFeatures, trainPlotFeatures, trainLabels, valVideoFeatures, valPlotFeatures, valLabels, merge_mode=argv[2])

def main_video():
    trainLabels, trainVideoFeatures = gather_features('train', return_plot=False)
    valLabels, valVideoFeatures = gather_features('val', return_plot=False)
 #   train_classifier_video(valVideoFeatures, valLabels)
    train_classifier_video(trainVideoFeatures, trainLabels, valVideoFeatures, valLabels)

def main_text():
    trainLabels, trainPlotFeatures = gather_features('train', return_video=False, reverse=True)
    valLabels, valPlotFeatures = gather_features('val', return_video=False, reverse=True)
    # to transpose the input, to get two lists of corresponding text & reversed
    trainPlotFeatures = np.array(map(list, zip(*trainPlotFeatures)))
    valPlotFeatures = np.array(map(list, zip(*valPlotFeatures)))
#    train_classifier_word_embedding(valPlotFeatures, valLabels)
    train_classifier_word_embedding(trainPlotFeatures, trainLabels, valPlotFeatures, valLabels)

if __name__=="__main__":
    from sys import argv
    code = argv[1]
    from time import time
    start = time()
    if code=='vl':
        main_vislang()
    if code=='v':
        main_video()
    if code=='t':
        main_text()
    if code=='ft':
        fine_tune_merge_only(argv[2])
    print time()-start,"seconds. Convert into days yourself :P"
