import cv2
import numpy as np
import sys,os
from glob import glob
from config.resources import video_resource
from config.global_parameters import frameWidth, frameHeight


def flip_frames(frames):
    return [cv2.flip(frame,1) for frame in frames]

def jitter_image(image):
    image = cv2.cvtColor(image, cv2.BGR2RGB)
    h,w,c = image.shape
    noise = np.random.randint(0,50,(h,w))
    zitter = np.zeros_like(image)
    channel = np.random.choice(3)
    zitter[:,:,channel] = noise  

    noise_added = cv2.add(img, zitter)
    jitteredImage = np.vstack((img[:h/2,:,:], noise_added[h/2:,:,:]))
    return jitteredImage 

def jitter_video(frames):
    return [jitter_image(frame) for frame in frames]

def get_frames(videoPath, start_time=1000, end_time=None, time_step=1000, return_gray=False):

    print "Getting frames for ",videoPath
    try:
        cap = cv2.VideoCapture(videoPath)

        if end_time == None:
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            total_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

            end_time = int(1000 * (total_frames/fps))
            
        for k in range(start_time, end_time+1, time_step):

            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, k)

            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (frameWidth, frameHeight))
                if return_gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield frame
    except Exception as e:
        print e
        return


def sequencify(videoFeatures, length=20, just_one=False):
    totalFrames = len(videoFeatures)
    for i in range(0, totalFrames, length):
        yield videoFeatures[i:i+length]
        if just_one:
            break
 


def gather_videos(genre, limit_videos = -1):
    """deprecated"""
    videoPaths = glob(video_resource+genre+'/*')
    videoFeatures = np.array([list(extract_feature_video(videoPath, verbose=True)) for videoPath in videoPaths[:limit_videos]])
    return videoFeatures


def save_frames_video(videoPath, videoID, outPath='./data'):

    if not os.path.exists(outPath):
        os.makedirs(outPath)
    
    if not os.path.exists(videoPath):
        print "Video not found"
        return None

    for frame_no, frame in enumerate(get_frames(videoPath)):
        frameWritePath = os.path.join(outPath,str(videoID)+'_'+str(frame_no)+'.jpg')
        print 'writing frame# {0} to {1}'.format(frame_no, frameWritePath)
        cv2.imwrite(frameWritePath, frame)


def get_videos(genre):
    
    videoPaths = glob(video_resource+genre+'/*')
    for videoID, videoPath in enumerate(videoPaths):
        print videoPath
        save_frames_video(videoPath, videoID, outPath='./data/'+genre)


if __name__=="__main__":
    get_videos(sys.argv[1])
