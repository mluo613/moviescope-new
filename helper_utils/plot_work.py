#vectorize plot

from pickle import load, dump
import numpy as np

from keras.optimizers import SGD
plotData = load(open('wiki_imdb_plot_labels.p', 'r'))

randomizedIndices = open('im_wiki_dataset.txt').read().split('\n')[:-1]
trainIndex, valIndex , testIndex = randomizedIndices
trainIndex = map(int,trainIndex.split(','))
valIndex = map(int, valIndex.split(','))
testIndex = map(int, testIndex.split(','))

print len(trainIndex), len(valIndex)

trainData = []
valData = []
testData = []

for index in trainIndex:
    key = plotData.keys()[index]
    data =\
    {'url':key,'index':index,'plot':plotData[key]['plot'],'movie_id':plotData[key]['movie_id'],'movie_title':plotData[key]['movie_title'],'genreLabel':plotData[key]['genreLabel']}
    trainData.append(data)

for index in valIndex:
    key = plotData.keys()[index]
    data =\
    {'url':key,'index':index,'plot':plotData[key]['plot'],'movie_id':plotData[key]['movie_id'],'movie_title':plotData[key]['movie_title'],'genreLabel':plotData[key]['genreLabel']}
    valData.append(data)

for index in testIndex:
    key = plotData.keys()[index]
    data =\
    {'url':key,'index':index,'plot':plotData[key]['plot'],'movie_id':plotData[key]['movie_id'],'movie_title':plotData[key]['movie_title'],'genreLabel':plotData[key]['genreLabel']}
    testData.append(data)

print trainData
print valData
print testData

from pickle import dump

dump(trainData, open('wiki_imdb_train.p','w'))
dump(valData, open('wiki_imdb_val.p','w'))
dump(testData, open('wiki_imdb_test.p','w'))

import sys
sys.exit()

xPlots, yLabels = [], []
for url in plotData:
    xPlots.append(plotData[url]['plot'])
    yLabels.append(plotData[url]['genreLabel'])

yLabels = np.array(yLabels)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()

x = count_vect.fit_transform(xPlots)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


vocab_size = X_train_tfidf.shape[1]

"""
print('Building model...')
model = Sequential()
model.add(Dense(4096, input_dim=vocab_size))
model.add(Activation('relu'))
model.add(Dense(2048, input_dim=vocab_size))
model.add(Activation('relu'))
model.add(Dense(1024, input_dim=vocab_size))
model.add(Activation('relu'))
model.add(Dense(512, input_dim=vocab_size))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256, input_dim=vocab_size))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(26))
model.add(Activation('sigmoid'))

learning_rate = 0.1
sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train_tfidf.toarray(), yLabels, nb_epoch=5, batch_size=32, validation_split=0.2)

prev_loss = 0
for i in range(25):
    history = model.fit(X_train_tfidf.toarray(), yLabels, nb_epoch=5, batch_size=32, validation_split=0.2)
    loss = np.mean(history.history['val_loss'])
    print loss, prev_loss
    if abs(loss - prev_loss) < 1e-4:
        if False:
            if learning_rate>1e-4:
                learning_rate /= 10
                print "Learning rate changed to:",learning_rate
                model.compile(loss='binary_crossentropy', optimizer=SGD(lr=learning_rate, momentum=0.9, nesterov=True), metrics=['accuracy'])
            else:
                break
        break

    prev_loss = loss
"""
model.save('first_text_model.h5')
