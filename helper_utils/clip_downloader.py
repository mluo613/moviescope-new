###############################################################################
# Author: Abhimanyu Banerjee
# Date Created: 1/30/2017
# 
# File Description: This script downloads the videos corresponding to each clip
# listed on the 'klipd.com' website
###############################################################################

from __future__ import print_function
from os.path import isfile, isdir, join
from multiprocessing import Pool
import json
import pdb
import os

'''downloads the media associated with the url passed as parameter'''
def downloadVideo(item):
    #pdb.set_trace()
    idx = item[0]
    video_url = item[1]["vid_url"]
    os.system("wget -O " + "../../klipd_data/clips/clip_" + str(idx) + ".mp4 " + video_url)

if __name__ == '__main__':
    
    pool = Pool(16)

    #relevant path names and urls
    video_json_fname = "vids_urls.json"
    klipd_vid_repo = "../../klipd_data"
    
    #check if data store for clip video urls exist
    if isfile(video_json_fname):
        with open(video_json_fname, "r") as f:
            video_dict = json.load(f)

            #check if the required repo directory for the videos exists
            if not isdir(klipd_vid_repo):
                os.mkdir(klipd_vid_repo)
                os.mkdir(join(klipd_vid_repo, "clips"))

            for item in video_dict["success"].items():
                downloadVideo(item)
            #pool.map(downloadVideo, video_dict["success"].items())
            