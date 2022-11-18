import os
from utils.data_utils import extract_images_from_video_dir


if __name__ == '__main__':
    cameras = ['C01', 'C02']

    for camera in cameras:
        video_dir = os.path.join('./data', camera, 'videos')
        output_dir = os.path.join('./data', camera, 'images')
        extract_images_from_video_dir(video_dir, output_dir)