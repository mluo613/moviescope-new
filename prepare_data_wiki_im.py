from utils import load_pkl, dump_pkl
from glob import glob
from config.resources import video_resource
from video import get_frames
from model_utils import get_features_batch

from text import WordEmbeddings, Text

# Read the wiki_im_{train,val,test} data with movie_id and plot
""" wiki_im_{} is a list of dictionaries with all relevant data like movie_id, plot text, and genres label """
# new_imdb5kpp_{}
""" get_{raw,features}_data add the path to the trailer path and/or extract features for video and plot and store it in additional pickles """

# baseName required only for this file. Not adding to config.
# wiki_imdb_{} is our root dataset. Will be saved with us until publication
# new_imdb5kpp_{}
_baseName = 'wiki_imdb_'
baseName = 'new_imdb5kpp_'

def _get_raw_data(mode='val'):

    rawData = [] #Include dictionaries containing trailer path, plot and genre labels & movie_id
    allData = load_pkl(baseName+mode)
    for data in allData:
        movie_id =  data['movie_id']
        path = glob(video_resource+str(movie_id)+'.*')[0]       
        plot = data['plot']
        genreLabel = data['genreLabel']

        rawData.append({'movie_id':movie_id,'plot':plot,'path':path,'genreLabel':genreLabel})

    dump_pkl(rawData, 'raw_data_'+mode)

def _get_features_data(mode='val'):
    """ deprecated with old dataset """
    """ Includes every sample with plotFeatures, videoFeatures, movie_id and genreLabel """
    featureData = []
    allData = load_pkl(baseName+mode)
    plots = []

    """Process plot vectors"""

    for data in allData:
        movie_id = data['movie_id']
        plot = data['plot']
        plots.append(plot)

    if mode=='train':
        textObj = Text()
        plotFeatures_all = textObj.fit_transform(plots)
        dump_pkl(textObj, 'plot_object_train')
    else:
        try:
            textObj = load_pkl('plot_object_train')
            plotFeatures_all = textObj.transform(plots).toarray()
        except:
            print "Please train the plots first."
            return

    plotIndex = -1
    for data in allData:
        plotIndex += 1
        movie_id =  data['movie_id']
        path = glob(video_resource+str(movie_id)+'.*')[0]       
        plot = data['plot']
        genreLabel = data['genreLabel']
        print plotIndex,"out of ",len(allData)
        print "Gathering features for",movie_id
        try:
            frames = list(get_frames(path, start_time=1000, end_time=200000, time_step=1000))
            videoFeatures = get_features_batch(frames, 'vgg16')
        except Exception as e:
            print e
            continue # Omit the movie if one of the feature is bad
            # videoFeatures = None
        plotFeatures = plotFeatures_all[plotIndex]

        featureData.append({'videoFeatures':videoFeatures, 'plotFeatures':plotFeatures, 'movie_id':movie_id, 'genreLabel':genreLabel})

    dump_pkl(featureData, 'feature_data_'+mode)

def old_main():
#    get_features_data('train')
#    get_features_data('val')
#    get_features_data('test')
    """ Dumps data to feature_data_{train,val,test """
    """ Use get_raw_data for miscellaneous experiments """
#    get_raw_data('val')
#    get_raw_data('train')
    get_raw_data('test')

def get_raw_data(mode='val'):

    rawData = [] #Include dictionaries containing plot and genre labels & movie_id
    allData = load_pkl(baseName+mode)
    for data in allData:
        movie_id =  data['movie_id']
        plot = data['plot']
        genreLabel = data['newGenreLabels']
        rawData.append({'movie_id':movie_id,'plot':plot,'newGenreLabels':genreLabel})

    dump_pkl(rawData, 'raw_data_'+mode)



def get_features_data(mode='val'):
    """ Includes every sample with plotFeatures, videoFeatures, movie_id and genreLabel """
    featureData = []
    allData = load_pkl(baseName+mode)
    plots = []

    """Process plot vectors"""

    for data in allData:
        movie_id = data['movie_id']
        plot = data['plot']
        plots.append(plot)

    if mode=='train':
        textObj = WordEmbeddings()
        plotFeatures_all = textObj.fit_transform(plots)
        dump_pkl(textObj, 'plot_object_train')
    else:
        try:
            textObj = load_pkl('plot_object_train')
            plotFeatures_all = textObj.transform(plots, reverse=False)
        except Exception as e:
            print e
            print "Please train the plots first."
            return

    plotIndex = -1
    for data in allData:
        plotIndex += 1
        movie_id =  data['movie_id']
        path = glob(video_resource+str(movie_id)+'.*')[0]       
        plot = data['plot']
        genreLabel = data['newGenreLabels']
        print plotIndex,"out of ",len(allData)
        print "Gathering features for",movie_id
        videoFeatures = data['videoFeatures']
        plotFeatures = plotFeatures_all[plotIndex]

        featureData.append({'videoFeatures':videoFeatures, 'plotFeatures':plotFeatures, 'movie_id':movie_id, 'genreLabel':genreLabel})

    dump_pkl(featureData, 'feature_data_'+mode)

def new_main():
    get_raw_data('test') 
    get_raw_data('train') 
    get_raw_data('val') 

new_main()
