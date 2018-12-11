from pandas import read_csv
import os

inputFile = read_csv('url.csv')
urls = inputFile['trailer_url'].values

from multiprocessing import Pool

p = Pool(16)

def download(id,url):
	print id,url
	os.system("youtube-dl https://www.youtube.com/watch?v="+url+" -o \"trailers/"+str(id)+".%(ext)s\"")

p.map(download, enumerate(urls))
