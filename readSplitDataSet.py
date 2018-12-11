import numpy as np
from pickle import load, dump

def load_pkl(pklName, verbose=True):
    if verbose:
        print ("Loading data from data/{0}.p".format(pklName))
    data = load(open(pklName+'.p', 'rb'), encoding='latin1')
    return data

def gather_raw_data(mode='train', return_plot=True, return_video=True, return_id=True):

    """ Returns plot features and/or video features and genre labels """
    print ("Gathering features from {} data".format(mode))
    pklFile = mode
    data = load_pkl(pklFile)
    plotFeatures, videoFeatures, labels = [], [], []
    ids = []
    for d in data:
        if return_id:
            ids.append(d['movie_id'])
        labels.append(d['newGenreLabels'])
        if return_plot:
            plotFeatures.append(d['plot'])
        #if return_video:
        #    videoFeatures.append(d['path'])


    labels = np.array(labels)
    return_list = [labels]

    if return_plot:
        plotFeatures = np.array(plotFeatures)
        return_list.append(plotFeatures)
    #if return_video:
    #    videoFeatures = np.array(videoFeatures)
    #    return_list.append(videoFeatures)
    if return_id:
        return_list.append(ids)

    return return_list

# trainList = gather_raw_data(mode='data/raw_data_train')
# valList = gather_raw_data(mode='data/raw_data_val')
# testList = gather_raw_data(mode='data/raw_data_test')

# print ('id', testList[2][0])
# print ('label', testList[0][0])
# print ('plot', testList[1][0])

# print ('id', testList[2][1])
# print ('label', testList[0][1])
# print ('plot', testList[1][1])