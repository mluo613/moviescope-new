""" Contains all useful functions """

import numpy as np
from pickle import load, dump


def sliding_window(image, windowSize, horizontalStride=4, verticalStride=4):

    """
    Runs a sliding window generator across the image 
        Inputs: 
            image: input image/frame
            windowSize: Tuple (width, height)
            stride: Window step size horizontal & vertical
        Output:
            Generator object with coordinates and the window cropped
    """
            
    for y in xrange(0, image.shape[0], verticalStride):
        for x in xrange(0, image.shape[1], horizontalStride):
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[0]])


def load_pkl(pklName, verbose=True):
    if verbose:
        print ("Loading data from {0}.p".format(pklName))
    data = load(open(''+pklName+'.p', 'rb'), encoding='latin1')
    return data


def dump_pkl(data, pklName, verbose = True):

    if verbose:
        print ("Dumping data into",pklName)
    dump(data, open('data/'+pklName+'.p', 'wb'))

def load_moviescope_model(modelName, verbose=True):

    from keras.models import load_model
    from config.resources import model_resource

    if modelName.find('h5')==-1:
        modelName+=".h5"
    if verbose:
        print ("Loading model:",modelName)
    model = load_model(model_resource+modelName)
    return model

def augment_labels(labels,videoFeatures):
    features,_labels = [],[]
    for label,videoFeature in zip(labels,videoFeatures):
        for feature in videoFeature:
            features.append(feature)
            _labels.append(label)

    features = np.array(features)
    labels = np.array(_labels)

    print (features.shape)
    print (labels.shape)
    return labels,features


def augment_labels_lstm(labels, videoFeatures, num_of_frames=9):

    from video import sequencify
    _labels,sequences = [], []
    for label,videoFeature in zip(labels,videoFeatures):
        for sequence in sequencify(videoFeature,num_of_frames, just_one=True):
            _labels.append(label)
            sequences.append(sequence)
            
    labels = np.array(_labels)
    features = np.array(sequences)

    num_of_samples = len(labels)
    input_dim = 4096 ## Change with different model
    dataTensor = np.zeros((num_of_samples,num_of_frames, input_dim))
    for sampleIndex in range(num_of_samples):
        for vectorIndex in range(num_of_frames):
            try:
                dataTensor[sampleIndex][vectorIndex]=features[sampleIndex][vectorIndex]
            except Exception as e:
                continue
 
    return labels,dataTensor

def gather_features(mode='train', return_plot=True, return_video=True, return_id=False, reverse=False):

    """ Returns plot features and/or video features and genre labels """
    print ("Gathering features from {} data".format(mode))
    pklFile = 'feature_data_'+mode
    data = load_pkl(pklFile)
    plotFeatures, videoFeatures, labels = [], [], []
    ids = []

    for d in data:
        if return_id:
            ids.append(d['movie_id'])
        if type(d['genreLabel']) == str:
            label = map(int,d['genreLabel'])
        else:
            label = d['genreLabel']
        labels.append(label)
        if False: # currently stores BagOfWords. Will edit later
            plotFeatures.append(d['plotFeatures'])
        if return_video:
            videoFeatures.append(d['videoFeatures'])


    labels = np.array(labels)
    return_list = [labels]

    if return_plot:
        if reverse:
            plotFeatures = load_pkl('plotFeatures_with_reverse_'+mode)
        else:
            plotFeatures = load_pkl('plotFeatures_'+mode)
#        plotFeatures = load_pkl('plotFeatures_with_reverse_'+mode)
        plotFeatures = np.array(plotFeatures)
        return_list.append(plotFeatures)
    if return_video:
        videoFeatures = np.array(videoFeatures)
        return_list.append(videoFeatures)
    if return_id:
        return_list.append(ids)

    return return_list

def gather_raw_data(mode='train', return_plot=True, return_video=True, return_id=False):

    """ Returns plot features and/or video features and genre labels """
    print( "Gathering features from {} data".format(mode))
    pklFile = 'raw_data_'+mode
    data = load_pkl(pklFile)
    plotFeatures, videoFeatures, labels = [], [], []
    ids = []
    for d in data:
        if return_id:
            ids.append(d['movie_id'])
        labels.append(d['genreLabel'])
        if return_plot:
            plotFeatures.append(d['plot'])
        if return_video:
            videoFeatures.append(d['path'])


    labels = np.array(labels)
    return_list = [labels]

    if return_plot:
        plotFeatures = np.array(plotFeatures)
        return_list.append(plotFeatures)
    if return_video:
        videoFeatures = np.array(videoFeatures)
        return_list.append(videoFeatures)
    if return_id:
        return_list.append(ids)

    return return_list
