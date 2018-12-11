from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import LSTM, Input, merge, Embedding, Lambda, Bidirectional
from utils import load_pkl
from bilinear_tensor import BilinearTensorLayer
from config.global_parameters import number_of_classes
from keras import backend as K
import numpy as np

def bilinear_projection(inputs):
    x, y = inputs
    batch_size = K.shape(x)[0]
    outer_product = x[:,:,np.newaxis] * y[:,np.newaxis,:]
    return K.reshape(outer_product,(batch_size, -1))


def remove_last_layers(model):
    """To remove the last FC layers of VGG and get the 4096 dim features"""
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []


vgg_model_16 = VGG16(include_top=True, weights="imagenet")
vgg_model_19 = VGG19(include_top=True, weights="imagenet")

remove_last_layers(vgg_model_16)
remove_last_layers(vgg_model_19)


def get_features_batch(frames, model_name="vgg16"):

    if model_name.lower() in ["vgg16", "vgg_16"]:
        model = vgg_model_16

    if model_name.lower() in ["vgg19", "vgg_19"]:
        model = vgg_model_19

    imageTensor = np.array(frames)

    ### /255 causing error. Maybe Vanishing gradients
    modelFeature =  model.predict(imageTensor, verbose=1)
    return modelFeature



def get_features(image, model_name="vgg16"):

    from keras import backend
    if backend.image_dim_ordering()=='th':
        print "Please switch to tensorflow backend. Update to reorder will come soon."
        return None

    if model_name.lower() in ["vgg16", "vgg_16"]:
        model = vgg_model_16

    if model_name.lower() in ["vgg19", "vgg_19"]:
        model = vgg_model_19

    imageTensor = np.zeros((1, 224, 224, 3))
    imageTensor[0] = image

    ### /255 causing error. Maybe Vanishing gradients
    modelFeature =  model.predict(imageTensor)[0]
    return modelFeature

def spatial_model(number_of_classes=2):
    """Classification layers here."""

    model = Sequential()
    model.add(Dense(2048, input_dim=4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(number_of_classes, activation='sigmoid'))

    return model

def good_vis_model(visInput, number_of_classes=number_of_classes, return_top=True, precompute=False):

    """ Bi-Directional LSTM as video features """

    print('Training Bi-directional LSTM for Video Features.')
    visModel = Bidirectional(LSTM(128, return_sequences=True))(visInput)
    visModel = Bidirectional(LSTM(64))(visModel)

    print 'merged fb'

    visModel = Dropout(0.25)(visModel)
#    visModel = Dense(64, activation='relu')(visModel)
    batchNormModel = BatchNormalization()(visModel)
#    visModel = Activation('relu')(batchNormModel)

    if return_top:
        visModel = Dropout(0.5)(visModel)
        outputLayer = Dense(number_of_classes, activation='sigmoid')(visModel)
        visModel = Model(input=[visInput], output=[outputLayer])

    return visModel


def vis_model(visInput, number_of_classes=number_of_classes, return_top=True, precompute=False):

    """ Bi-Directional LSTM as video features """

    print('Training Bi-directional LSTM for Video Features.')
    visModel_f = LSTM(128, return_sequences=True)(visInput)
    visModel_f = LSTM(64)(visModel_f)

    print 'f initialized'

    visModel_b = LSTM(128, return_sequences=True, go_backwards=True)(visInput)
    visModel_b = LSTM(64)(visModel_b)

    print 'b initialized'

    visModel = merge([visModel_f, visModel_b], mode='concat')

    print 'merged fb'

    visModel = Dropout(0.25)(visModel)
#    visModel = Dense(64, activation='relu')(visModel)
    visModel = Dense(64)(visModel)
    batchNormModel = BatchNormalization()(visModel)
#    visModel = Activation('relu')(batchNormModel)

    if return_top:
        visModel = Dropout(0.5)(visModel)
        outputLayer = Dense(number_of_classes, activation='sigmoid')(visModel)
        visModel = Model(input=[visInput], output=[outputLayer])

    return visModel


def better_text_model(sequence_input=None, number_of_classes=number_of_classes,
return_top=True, use_embedding=True, trainable=False):

    if sequence_input==None:
        print "No input received."
        return

    textObj = load_pkl('plot_object_train')
    if use_embedding:
        _weights = [textObj.embedding_matrix]
    else:
        print "not using Embeddings"
        _weights = None
    print('Training Bi-directional LSTM Word Embedding model.')
    embedding_layer = Embedding(textObj.num_words+1, textObj.embeddingDim, weights=_weights,
    input_length=textObj.maxSequenceLength, trainable=trainable)

    embedded_sequences = embedding_layer(sequence_input)

    bidir = Bidirectional(LSTM(128, return_sequences=True))(embedded_sequences)
    bidir = Bidirectional(LSTM(64))(bidir)
    print 'merged fb'
    textModel = Dropout(0.25)(bidir)
#    textModel = Dense(64, activation='relu')(textModel)
#    textModel = Dense(64)(textModel)
    textModel = BatchNormalization()(textModel)
#    textModel = Activation('relu')(textModel)

    if return_top:
        textModel = Dropout(0.5)(textModel)
        outputLayer = Dense(number_of_classes, activation='sigmoid')(textModel)
        textModel = Model(input=[sequence_input, sequence_input_reverse],  output=outputLayer)
    return textModel 


def good_text_model(sequence_input=None, sequence_input_reverse=None, number_of_classes=number_of_classes,
return_top=True, use_embedding=True, trainable=False):

    if sequence_input==None:
        print "No input received."
        return

    textObj = load_pkl('plot_object_train')
    if use_embedding:
        _weights = [textObj.embedding_matrix]
    else:
        print "not using Embeddings"
        _weights = None
    print('Training Bi-directional LSTM Word Embedding model.')
    embedding_layer = Embedding(textObj.num_words+1, textObj.embeddingDim, weights=_weights,
    input_length=textObj.maxSequenceLength, trainable=trainable)

    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences_reverse = embedding_layer(sequence_input_reverse)

    forward = LSTM(128, return_sequences=True)(embedded_sequences)
    forward = LSTM(64)(forward)

    print 'f initialized'

    backward = LSTM(128, return_sequences=True)(embedded_sequences_reverse)
    backward = LSTM(64)(backward)

    print 'b initialized'

    bidir = merge([forward,backward],mode='concat')

    print 'merged fb'
    textModel = Dropout(0.25)(bidir)
#    textModel = Dense(64, activation='relu')(textModel)
    textModel = Dense(64)(textModel)
    textModel = BatchNormalization()(textModel)
#    textModel = Activation('relu')(textModel)

    if return_top:
        textModel = Dropout(0.5)(textModel)
        outputLayer = Dense(number_of_classes, activation='sigmoid')(textModel)
        textModel = Model(input=[sequence_input, sequence_input_reverse],  output=outputLayer)
    return textModel 


def text_model(sequence_input=None, number_of_classes=number_of_classes, input_dim=None, return_top=True):

    if sequence_input==None:
        print "No input received."
        return

    textObj = load_pkl('plot_object_train')
    print('Training Bi-directional LSTM Word Embedding model.')
    embedding_layer = Embedding(textObj.num_words+1, textObj.embeddingDim, weights=[textObj.embedding_matrix], input_length=textObj.maxSequenceLength, trainable=False)

    embedded_sequences = embedding_layer(sequence_input)

    forward = LSTM(128, return_sequences=True)(embedded_sequences)
    forward = LSTM(64)(forward)

    print 'f initialized'

    backward = LSTM(128, go_backwards=True, return_sequences=True)(embedded_sequences)
    backward = LSTM(64)(backward)

    print 'b initialized'

    bidir = merge([forward,backward],mode='concat')

    print 'merged fb'
    textModel = Dropout(0.25)(bidir)
#    textModel = Dense(64, activation='relu')(textModel)
    textModel = Dense(64)(textModel)
    textModel = BatchNormalization()(textModel)
    textModel = Activation('relu')(textModel)

    if return_top:
        textModel = Dropout(0.5)(textModel)
        outputLayer = Dense(number_of_classes, activation='sigmoid')(textModel)
        textModel = Model(input=sequence_input,  output=outputLayer)
    return textModel 


def vislang_model(merge_mode, return_top=True):

    input_dim = 4096
    number_of_frames = 200

    ## Text modelling

    sequence_input = Input(shape=(3000,), dtype='int32')
    sequence_input_reverse = Input(shape=(3000,), dtype='int32')
    textModel = good_text_model(sequence_input, sequence_input_reverse, return_top=False)

    ## video modelling

    visInput = Input(shape=(number_of_frames, input_dim), dtype='float32')
    visModel = vis_model(visInput, number_of_classes, return_top=False)
    if merge_mode == 'bilinear':
        predictionLayer = BilinearTensorLayer(input_dim=64)([visModel, textModel])
    else:
        if merge_mode in ['concat','sum','mul']:
            vislangModel = merge([visModel, textModel], mode=merge_mode, name='vislang')
        else:
            vislangModel = Lambda(bilinear_projection, output_shape=(4096,))([visModel, textModel])
    #        vislangModel = Dense(1024, activation='relu')(vislangModel)

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
        """
        else:
            if merge_mode in ['concat','sum','mul']:
                vislangModel = merge([visModel, textModel], mode=merge_mode, name='vislang')
               

            elif merge_mode=='outer':
                vislangModel = Lambda(bilinear_projection, output_shape=(4096,))([visModel, textModel])
                vislangModel = Dense(1024, activation='relu')(vislangModel)

            vislangModel = Dense(64, activation='relu')(vislangModel)
        """
        vislangModel = Dropout(0.25)(vislangModel)
        predictionLayer = Dense(number_of_classes, activation='sigmoid', name='main_output')(vislangModel)
    
    model = Model(input=[visInput, sequence_input, sequence_input_reverse], output=[predictionLayer])

    return model

    
if __name__=="__main__":
    import cv2
    inputImage = cv2.resize(cv2.imread("testImages/test1.jpg"), (224, 224))
    from time import time
    start = time()
    vector = get_features(inputImage, 'vgg19')
    print 'time taken by vgg 19:',time()-start,'seconds. Vector shape:',vector.shape
    start = time()
    vector = get_features(inputImage, 'vgg16')
    print 'time taken by vgg 16:',time()-start,'seconds. Vector shape:',vector.shape

    model = spatial_model(4)
    print model.summary()
