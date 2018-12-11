import pandas as pd
from csv import writer
import urllib
import urllib2
from bs4 import BeautifulSoup


data = pd.read_csv('data/movie_metadata.csv')

movie_list = data['movie_title'].values

def get_best_url(name):
    query = urllib.quote(name+' trailer')
    url = "https://www.youtube.com/results?search_query=" + query
    response = urllib2.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, "lxml")
    for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
        url = 'https://www.youtube.com' + vid['href']
        return url

with open('url.csv', 'a') as outFile:
    writer = writer(outFile)
    for index in range(1257, 1500):#len(movie_list)):
        if index==-1:
            writer.writerow(['id','movie_title','trailer_url'])
        movie_title = movie_list[index]
        print movie_title,
        trailer_url = get_best_url(movie_title)
        video_id = trailer_url.split('watch?v=')[-1]
        writer.writerow([index, movie_title, video_id])
        print index,trailer_url


