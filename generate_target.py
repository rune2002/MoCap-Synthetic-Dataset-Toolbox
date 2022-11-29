"""Generate target files. This is only for HKMC users"""
import os
import json
import numpy as np

from utils.data_utils import uni_to_pose, save_npz
from utils.camera import mocap_to_camera, extrinsic_parameter

if __name__ == '__main__':
    
    cameras = ['C01', 'C02']
    with open('./data/config.json', 'r') as f:
        data = json.load(f)

    MOCAP_TO_HKMC = data["MOCAP_TO_HKMC"]
    tc = np.array([-56.98, 129.79, -0.38])
    R, t = extrinsic_parameter(1.34, -1.47, -42.30, tc)

    for camera in cameras:
        d = dict()
        kps3d_dir = os.path.join('./data', camera, 'kps3d')
        output_dir = os.path.join('./data', camera, 'target.npz')
        for file in os.listdir(kps3d_dir):
            _, filename = os.path.split(file)
            pose = uni_to_pose(os.path.join(kps3d_dir, file), MOCAP_TO_HKMC)
            tc = np.array([-56.98, 129.79, -0.38])
            R, t = extrinsic_parameter(1.34, -1.47, -42.30, tc)
            pose = [mocap_to_camera(p, R, t) for p in pose] 
            d[filename] = pose

        save_npz(os.path.join('./data', camera, 'target.npz'), d)