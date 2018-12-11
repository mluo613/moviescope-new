from utils import load_pkl, augment_labels_lstm, gather_features, load_moviescope_model, gather_raw_data, dump_pkl
from sys import exit, argv
import numpy as np
from config.resources import best_plot_model, best_video_model
#from keras.metrics import precision, recall, fmeasure
from sklearn.metrics import *
from model_utils import vislang_model


def evaluate_visual(model, valFeatures, valLabels):
    results = []
    Labels, Features = augment_labels_lstm(valLabels, valFeatures, 200)
    print Labels.shape, Features.shape
    print "Validation loss and accuracy for Videos."
    print model.evaluate(Features, Labels)


def evaluate_text(model, Features, Labels):

    print Labels.shape, Features.shape
    print "Validation loss and accuracy for Plots."
    print model.evaluate(Features, Labels)

def test_visual():
    valLabels, valFeatures = gather_features('val', return_plot=False)
    model = load_moviescope_model(best_video_model)
    evaluate_visual(model, valFeatures, valLabels)

def test_text():
    valLabels, valFeatures = gather_features(mode='val', return_video=False)
    model = load_moviescope_model(best_plot_model)
    evaluate_text(model, valFeatures, valLabels)

def test(mode='val'):
    valLabels, plotFeatures, videoFeatures = gather_features(mode)
    plotModel = load_moviescope_model(best_plot_model)
    videoModel = load_moviescope_model(best_video_model)
    evaluate_text(plotModel, plotFeatures, valLabels)
    evaluate_visual(videoModel, videoFeatures, valLabels)

def evaluate_vislang(model, videoFeatures, plotFeatures, labels):
    Labels, videoFeatures = augment_labels_lstm(labels, videoFeatures, 200)
    print model.evaluate([videoFeatures, plotFeatures], Labels)
    
def test_vislang(mode='val'):
    model = load_moviescope_model('wiki_im_vislang')
    labels, plotFeatures, videoFeatures = gather_features(mode)
    evaluate_vislang(model, videoFeatures, plotFeatures, labels)

def generate_precision_recall_video(mode='val'):

    model = load_moviescope_model('wiki_im_video_sgd')
    yTrue, videoFeatures = gather_features(mode, return_plot=False)
    _, videoFeatures = augment_labels_lstm(yTrue, videoFeatures, 200)
    yPreds = model.predict(videoFeatures)

    dump_pkl((yTrue, yPreds), mode+'_pred_video_sgd')

    return

def generate_precision_recall_text(mode='val'):

    model = load_moviescope_model('text')
    yTrue, plotFeatures = gather_features(mode, return_video=False, reverse=True)
    plotFeatures = np.array(map(list, zip(*plotFeatures)))
    yPreds = model.predict([plotFeatures[0], plotFeatures[1]])
    dump_pkl((yTrue, yPreds), mode+'_pred_text')

    return

def generate_precision_recall_vislang(mode='val', merge_mode='sum'):

    if merge_mode == 'bilinear':
        model = vislang_model(merge_mode)
        model.load_weights('data/weights/weights_min_loss_%s.h5' % merge_mode)
    else:
        model = load_moviescope_model('eq_VisLang_%s' % merge_mode)

    yTrue, plotFeatures, videoFeatures = gather_features(mode, reverse=True)
    plotFeatures = np.array(map(list, zip(*plotFeatures)))
    _, videoFeatures = augment_labels_lstm(yTrue, videoFeatures, 200)
    yPreds = model.predict([videoFeatures, plotFeatures[0], plotFeatures[1]])
    dump_pkl((yTrue, yPreds), mode+'_pred_eq_vislang_'+merge_mode)


def return_confident_results(mode='val'):
    
    model = load_moviescope_model('wiki_im_VisLang')
    genrePredictionDict = dict((i,[]) for i in range(26))
    textObj = load_pkl('plot_object_train')
    labels, plotFeatures, videoFeatures, movieIds = gather_features(mode, return_id=True)
    _, videoFeatures = augment_labels_lstm(labels, videoFeatures, 200)
    predictionScores = model.predict([videoFeatures, plotFeatures])
    for index  in range(len(predictionScores)):
        for i in range(26):
            genrePredictionDict[i].append((predictionScores[index][i],movieIds[index]))

    dump_pkl(genrePredictionDict, 'genrePredictionDict_'+mode)
    
    for i in range(26):
        print sorted(genrePredictionDict[i], reverse=True)[:10]
    return


if __name__=="__main__":
    generate_precision_recall_vislang('val', merge_mode=argv[1])
