import sys

import cv2
import mediapipe as mp

from sample import mp_sample

VIDEO_PATH_LIST = [
    'data/20230514192027.mp4',
    'data/IMG_9417.MOV',
]


def main(video_path):
    mp_sample(video_path)


if __name__ == '__main__':
    _movie_index = int(sys.argv[1])
    main(VIDEO_PATH_LIST[_movie_index])
