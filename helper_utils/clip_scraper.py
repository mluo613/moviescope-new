###############################################################################
# Author: Abhimanyu Banerjee
# Date Created: 1/24/2017
# 
# File Description: This script scrapes a website hosting movie clips to collect 
# a dataset of clip urls and related information about the clip and the movie 
# it is associated with.
###############################################################################

from __future__ import print_function
from bs4 import BeautifulSoup
import urllib
import pandas as pd
import json
from os.path import isfile
import requests
import pdb

TOKEN_SECRET = "124983501" #insert your diffbot.com developer secret here

'''parse the home page of the website of interest and extract the links of the
webpages for each genre listed in the navigation bar of the website'''
def getGenreLinks(root_url):
    print("\nScraping website: ", root_url)
    
    #establish connection to the root(home) url and generate its soup
    print("\nScraping root url..")
    html = urllib.request.urlopen(root_url)
    soup = BeautifulSoup(html.read(), 'html.parser')
    print("Root url scraped!")

    #find the dropdowns and then the genres which are in the first dropdown menu
    dropdown_soup = soup.findAll(attrs={'class':'yamm-content'})
    genre_soup = dropdown_soup[0].findAll("a")
    
    #extract link for each genre
    genre_list = []
    for genre in genre_soup:
        genre_list.append(genre["href"])
    return genre_list

'''parse webpage associated with each genre url and get the urls for each movie
in each genre'''
def getMovieLinks(genre_url_list):
    
    clips_url_list = {}
    #cycle through the links for each genre and scrape the page for clip urls
    for genre_url in genre_url_list:
        genre = genre_url.split("/").pop()
        
        print("\nScraping the {} genre web page..".format(genre))
        genre_html = urllib.request.urlopen(genre_url)
        movie_soup = BeautifulSoup(genre_html.read(), 'html.parser') 
        
        #extract the page loader button and get the number of pages in the
        #current genre webpage
        num_pages = 1
        loader_buttons = movie_soup.findAll(attrs={'id':'loadMore'})
        if len(loader_buttons) != 0:
            loader_button = loader_buttons[0]
            num_pages = int(loader_button.attrs['data-totalpages'])

        num_genre_clips = 0
        genre_clip_urls = []
        #iterate over the number of pages the clips are spread over
        for idx in range(num_pages):
            
            #send an ajax request to fetch the next page of clip urls for 
            #current genre
            data = {'page': idx, 'action': 'ajax'}
            page = requests.get(genre_url, data)    
            
            #convert the ajax response to a Soup and scrape it for the divs 
            #containing the anchor tags with the clip urls
            page_soup = BeautifulSoup(page.text, 'html.parser')
            poster_list = page_soup.findAll("div", attrs={'class':'poster'})
            poster_links = [str(poster).split('a href="')[1].split('"')[0] for poster in poster_list]
            
            num_genre_clips += len(poster_links)
            print(num_genre_clips, "clips from {} genre".format(genre), end="\r",)
            genre_clip_urls.append(poster_links)
            
        print("{} genre page scraping complete!".format(genre.capitalize()))
        clips_url_list[genre] = [clip_url for clip_list in genre_clip_urls for clip_url in clip_list]

    #sanity check to see how many clips from each genre was scraped
    for key, value in clips_url_list.items():
        print("{0}: {1} clips\t".format(key, len(value)), end="\r",)

    return clips_url_list


def getVideoUrls(clip_url_list):

    num_clips, num_videos = 0, 0
    #setup for using scraping api
    api_root_url = "https://api.diffbot.com/v3/video?token=" + TOKEN_SECRET
    
    video_dict = { "success" : {}, "failures" : {}}
    for genre, url_list in clip_url_list.items():
        retry_urls = [] #list of urls that need age verification
        for url in url_list:
            num_clips += 1

            #send diffbot video api request to scrape for mp4 links in current url
            request_url = api_root_url + "&url=" + url
            with urllib.request.urlopen(request_url) as response:
                #pdb.set_trace()
                api_response = response.read()
                try:
                    video_url = str(api_response).split("src=\\\\")[1].split("\\\\")[0][1:]
                    num_videos += 1
                    video_dict["success"][str(num_videos)] = { "page_url" : url, "vid_url" : video_url}
                    print("{0} video urls scraped out of {1} clip pages".format(num_videos, num_clips), end="\r",)
                except IndexError:
                    retry_urls.append(url)

        if len(retry_urls) != 0:
            video_dict["failures"][genre] = retry_urls #maintain a dict of urls to retry later
    return video_dict


def getClipData(clip_dict):
    clip_descr_text, movie_descr_text = "", ""
    num_pages = 0
    for idx, clip_data in clip_dict.items():
        url = clip_data["page_url"]
        
        #scrape the current url for other metadata
        clip_html = urllib.request.urlopen(url)
        clip_soup = BeautifulSoup(clip_html.read(), 'html.parser')
        
        #extract the items of interest
        release_year = clip_soup.find(attrs={'class':'movieReleaseTag'}).text.split(": ").pop()
        movie_name = clip_soup.find(attrs={'class':'mname'}).text
        descr_tag = clip_soup.find(attrs={'id':'clip-description'})
        info_tags = descr_tag.findAll("dl")
        clip_descr_list = info_tags[0].findAll("dd")
        for clip_descr in clip_descr_list:
            clip_descr_text += clip_descr.text
        movie_descr_list = info_tags[1].findAll("dd")
        for movie_descr in movie_descr_list:
            movie_descr_text += movie_descr.text

        #populate the clip dict with the data scraped
        clip_data["release_date"] = release_year
        clip_data["movie_name"] = movie_name
        clip_data["clip_descr"] = clip_descr_text
        clip_data["movie_descr"] = movie_descr_text
        clip_dict[idx] = clip_data

        num_pages += 1
        print("{0} pages scraped".format(num_pages), end="\r",)

    return clip_dict

def convertToCSV(clip_data_json):

    klipd_csv_path = "klipd_data.csv"
    with open(klipd_csv_path, "w"):
        #pdb.set_trace()
        print("\nConverting 'klipd' data from json to csv..")
        clip_data_df = pd.DataFrame(clip_data_json).transpose()
        #convert dataframe index to integers
        clip_data_df.index = clip_data_df.index.map(int) 
        clip_data_df.sort_index(axis=0, inplace=True)
        clip_data_df.to_csv(klipd_csv_path)
        print("'Klipd' data converted from json to csv and stored!")



if __name__ == '__main__':
    
    #relevant path names and urls
    url = "http://klipd.com/"
    clip_json_fname = "clip_page_urls.json"
    video_json_fname = "vids_urls.json"
    clip_dataset_fname = "clip_dataset.json"

    #check if data store for the 'klipd' dataset exists, therwise create it
    if not isfile(clip_dataset_fname):

        #check if data store for the urls of the videos for each clip exists,
        #otherwise create one
        if not isfile(video_json_fname):

            #check if data store for the urls of each clip detail page exists,
            #otherwise create one
            if not isfile(clip_json_fname):

                #creating data store for urls of each clip detail page
                print("\nNo data store for clips url list exist. Building data store..")
                with open(clip_json_fname, "w") as f:
                    genre_url_list = getGenreLinks(url)
                    clip_url_list = getMovieLinks(genre_url_list)
                    json.dump(clip_url_list, f)
                print("Data store clips url list built!")

            #data store for the urls of each clip detail page exists
            else:
                print("\nLoading list of clip urls..")
                with open(clip_json_fname, "r") as f:
                    clip_url_list = json.load(f)
                print("Clip urls list loaded!")

            #creating data store for the urls of the videos for each clip
            print("\nNo data store for video urls exist. Building data store..")
            with open(video_json_fname, "w") as f:
                vid_dict = getVideoUrls(clip_url_list)
                json.dump(vid_dict, f)
            print("List of video URLs scraped and data store built!")

        #data store for the urls of the videos for each clip exists
        else:
            print("\nLoading list of video urls..")
            with open(video_json_fname, "r") as f:
                vid_dict = json.load(f)
            print("List of video urls loaded!")

            #creating data store for the 'klipd' dataset
            print("\nCreating final klipd dataset..")
            clip_dataset = getClipData(vid_dict["success"])
            with open(clip_dataset_fname, "w") as f:
                json.dump(clip_dataset, f)
            print("Klipd dataset creation complete!")
    
    #data store for the 'klipd' data exists
    else:
        print("\nLoading Klipd dataset ..")
        with open(clip_dataset_fname, "r") as f:
            klipd_dataset = json.load(f)
        print("Klipd dataset loaded!")
        klipd_df = convertToCSV(klipd_dataset)