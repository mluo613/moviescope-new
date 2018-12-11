from bs4 import BeautifulSoup
import urllib2
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
from csv import writer

data = pd.read_csv('data/movie_metadata.csv')
imdb_page = data['movie_imdb_link'].values

with open('summary_storyline.csv','a') as outFile:
    writer = writer(outFile) 
    for index in range(-1,len(imdb_page)):
        if index==-1:
            writer.writerow(['id','movie_imdb_link','summary','storyline'])
            continue
            
        url = imdb_page[index]
        print url,
            
        response = urllib2.urlopen(url)
        html = response.read()
        soup = BeautifulSoup(html, "lxml")

        summary_soup = soup.findAll(attrs={'class':'summary_text'})
        storyline_soup = soup.findAll(attrs={'class':'inline canwrap'})

        if summary_soup == None or summary_soup == []:
            summary = ""
        else:
            summary = summary_soup[0].get_text().strip()

        if storyline_soup == None or storyline_soup == []:
            storyline = ""
        else:
            storyline = storyline_soup[0].get_text().strip().split('Written by')[0].strip()

        print summary, storyline
        writer.writerow([index,url,summary,storyline])
        print index
        print "*"*20


