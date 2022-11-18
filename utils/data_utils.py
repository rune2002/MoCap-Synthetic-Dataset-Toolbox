import os
import re
import numpy as np
import cv2


def save_npz(filename, d):
    """Save dictionary to npz format"""
    # TODO Add Data Assert
    _, ext = os.path.splitext(filename)
    assert ext == '.npz'
    np.savez(filename, **d)


def load_npz(npz):
    """Load npz to dictionary format"""
    # TODO Add Data Assert
    d = dict(np.load(npz, allow_pickle=True))
    return d


def generate_npz(dir):
    """Generate npz file from uni folder
    This is only for HKMC users"""
    d = dict()
    for file in os.listdir(dir):
        p = os.path.join(dir, file)
        id, ext = os.path.splitext(file)
        if ext == '.uni':
            d[id] = uni_to_pose(p)
    return d


def uni_to_pose(uni, MOCAP_TO_HKMC, downsample=3):
    """Extract pose data from uni format
    This is only for HKMC users"""

    with open(uni) as f:
        lines = f.read().splitlines()

    p = re.compile('Frame')
    q = re.compile('-?\d+\.?\d*')
    r = re.compile('b\d{2}')

    start, end = q.findall(lines[0])
    joints = r.findall(lines[1])
    
    data = []
    d = []
    for line in lines[2:]:
        if p.match(line):
            continue
        
        xyz = list(map(float, q.findall(line)))
        if len(xyz) == 3:
            d.append(xyz)
        
        if len(d) == len(joints):
            d.append([(d[21][i] + d[24][i]) / 2.0 for i in range(3)])
            d.append([(d[12][i] + d[16][i]) / 2.0 for i in range(3)])
            data.append(d.copy())
            d.clear()
    
    data = data[::downsample]
    data = np.take(np.array(data), MOCAP_TO_HKMC, axis=1)
    return data


def extract_images_from_video_dir(video_dir, output_dir='./_output', downsample=3):
    videos = os.listdir(video_dir)
    for video in videos:
        extract_images_from_video(os.path.join(video_dir, video), output_dir, downsample)
        

def extract_images_from_video(video, output_dir, downsample=3):
    _, filename = os.path.split(video)
    name, ext = os.path.splitext(filename)
    assert ext == '.mp4'
    images_path = os.path.join(output_dir, name)

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    vc = cv2.VideoCapture(video)
    success, image = vc.read()
    frame = 0
    while success:
        cv2.imwrite(os.path.join(images_path, f'{frame//downsample:03d}.png'), image)
        frame += downsample
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = vc.read()