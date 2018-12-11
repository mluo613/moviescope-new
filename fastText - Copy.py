
import numpy as np, sys
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import GlobalAveragePooling1D, Dropout
from keras.datasets import imdb
from keras.optimizers import Adam


from utils import load_pkl, gather_features, dump_pkl


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indices, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i+ngram_value])
                if ngram in token_indices:
                    new_list.append(token_indices[ngram])
        new_sequences.append(new_list)
    return new_sequences

ngram_range = 1
#max_features = 25000
maxlen = 3000
batch_size = 32
embedding_dims = 300

yData = load_pkl('data/plotFeatures_with_reverse_val')
y_test = load_pkl('data/valLabels')
x_test = [d[0][0000:].tolist() for d in yData]

yData = load_pkl('data/plotFeatures_with_reverse_train')
y_train = load_pkl('data/trainLabels')
x_train = [d[0][0000:].tolist() for d in yData]

yData = load_pkl('data/plotFeatures_with_reverse_test')
y_final_test = load_pkl('data/testLabels')
x_final_test = [d[0][0000:].tolist() for d in yData]

mf = 0
for i in x_train:
    m = max(i)
    if m>mf:
        mf=m
for i in x_test:
    m = max(i)
    if m>mf:
        mf=m
for i in x_final_test:
    m = max(i)
    if m>mf:
        mf=m

max_features = mf+1
#(x_train, y_train), (x_test, y_test) = imdb.load_data()
print()
print ('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print ('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
print ('Average test sequence length: {}'.format(np.mean(list(map(len, x_final_test)), dtype=int)))

if ngram_range > 1:
    print( 'Adding {}-gram features'.format(ngram_range))

    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range+1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)
    start_index = max_features + 1
    token_indices = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indices[k]: k for k in token_indices}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indices, ngram_range)
    x_test = add_ngram(x_test, token_indices, ngram_range)
    x_final_test = add_ngram(x_final_test, token_indices, ngram_range)
    print( 'Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print( 'Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
    print( 'Average test sequence length: {}'.format(np.mean(list(map(len, x_final_test)), dtype=int)))

#print 'Pad sequences (samples x time)'
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#sys.exit()
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

x_train, x_test = np.array(x_train), np.array(x_test)
x_final_test = np.array(x_final_test)
print ('x_train shape:', x_train.shape)
print ('x_test shape:', x_test.shape)
print ('x_test shape:', x_final_test.shape)

print ('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
print (x_test[0])
print (max_features)
embeddingWeights = load_pkl('data/glove_embeddings/glove_weights')

#input_layer = Input(shape=(1,), dtype='int32')

# embedding = Embedding(max_features+1,
#                     embedding_dims,
#                     input_length=maxlen,
#                     weights=[embeddingWeights])

embedding = Embedding(max_features+1,
                    embedding_dims,
                    input_length=maxlen)
model.add(embedding)

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())
# We project onto a single unit output layer, and squash it with a sigmoid:
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(13, activation='sigmoid'))

adam = Adam(lr=0.01, decay=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#model.load_weights('data/weights/fasttext_uni.h5')
#model.load_weights('data/glove_embeddings/best_weights_glove.h5')
#scores = model.predict(x_train)
#dump_pkl((y_train, scores), 'train_pred_fasttext')
from keras.callbacks import ModelCheckpoint, RemoteMonitor

print (max_features)
#remote = RemoteMonitor(root='http://128.143.63.199:9009')
#check_glove = ModelCheckpoint(filepath='data/glove_embeddings/best_weights_glove.h5',verbose=1, save_best_only=True, monitor='val_acc')
#check = ModelCheckpoint(filepath='data/weights/fasttext_uni_64_drop.h5', save_weights_only=True, mode='max',
#monitor='val_acc', save_best_only=True, verbose=1)
epochs=150
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(x_test, y_test))

scores = model.evaluate(x_final_test, y_final_test, batch_size=batch_size)

print("Accuracy: ", str(scores[1]))