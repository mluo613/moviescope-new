from csv import reader, writer
from sys import argv
import urllib
from bs4 import BeautifulSoup
from urllib2 import urlopen
rows = []

summaryFile = 'plot_summaries.txt'
movieFile = 'movie.metadata.tsv'

def read_tsv(fileName):
    with open(fileName) as tsvin:
        tsvData = reader(tsvin, dialect='excel-tab')
        for row in tsvData:
            yield row


summaryData = list(read_tsv(summaryFile))
movieData = list(read_tsv(movieFile))

dataDict = dict([(row[0],{'plot':row[1]}) for row in summaryData])

for movie in movieData:
    _id = movie[0]
    if dataDict.has_key(_id):
        dataDict[_id]['movie_title']=movie[2]


def get_url(_id):
    movie_title = dataDict[_id]['movie_title']
    query = urllib.quote(movie_title+' trailer')
    url = "https://www.youtube.com/results?search_query=" + query
    response = urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, "lxml")
    for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
        url = 'https://www.youtube.com' + vid['href']
        break
    url = url.split('watch?v=')[1][:11]
    return url

with open('url_42k_trailer.csv','a') as outFile:
    out = writer(outFile)
    out.writerow(['id','movie_title','trailer_id','first_link_error'])
    for _id in dataDict.keys():
        error = False
        try:
            movie_title = dataDict[_id]['movie_title']
            query = urllib.quote(movie_title+' trailer')
            url = "https://www.youtube.com/results?search_query=" + query
            response = urlopen(url)
            html = response.read()
            soup = BeautifulSoup(html, "lxml")
            for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
                url = 'https://www.youtube.com' + vid['href']
                try:
                    url = url.split('watch?v=')[1][:11]
                    break
                except Exception as e:
                    error = e 
                    continue
            print _id,dataDict[_id]['movie_title'], url
            out.writerow([_id,dataDict[_id]['movie_title'],url,error])
        except Exception as e:
            error = True
            print _id,e
            out.writerow([_id,'','',e])
