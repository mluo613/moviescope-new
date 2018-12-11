from csv import reader, writer
from pandas import read_csv
from sys import argv

imdbFile = '../url_final_5042.csv'
wikiFile = 'url_42k_trailer.csv'
summaryFile = 'plot_summaries.txt'
movieFile = 'movie.metadata.tsv'


def read_tsv(fileName):
    with open(fileName) as tsvin:
        tsvData = reader(tsvin, dialect='excel-tab')
        for row in tsvData:
            yield row

data1 = read_csv(imdbFile)
imdbIds = set(data1['trailer_url'].values)
data = read_csv(wikiFile)
wikiIds = set(data['trailer_id'].values)
movie_ids = data['id'].values


commonIds = list(wikiIds.intersection(imdbIds))

summaryData = list(read_tsv(summaryFile))
movieData = list(read_tsv(movieFile))

dataDict = dict([(row[0],{'plot':row[1]}) for row in summaryData])

for movie in movieData:
    _id = movie[0]
    if dataDict.has_key(_id):
        dataDict[_id]['movie_title']=movie[2]

wiki_trailer_dict = {}
movie_ids = map(str,movie_ids)


wiki_imdb_trailer = {}
for d in data.values:
    if d[2] in commonIds:
        wiki_imdb_trailer[d[2]] = dataDict[str(d[0])]
        #wiki_imdb_trailer[d[2]]['movie_id'] = str(d[0])

for d in data1.values:
    if d[2] in commonIds:
        wiki_imdb_trailer[d[2]]['movie_id'] = d[0]
print wiki_imdb_trailer


import pickle as p

p.dump(wiki_imdb_trailer,open('wiki_imdb_trailer_plot.p','w'))

