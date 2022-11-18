import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from skeleton import HKMC_ADJ_LIST
from viz import show_3d_pose
from data_utils import uni_to_pose
from camera import mocap_to_camera, extrinsic_parameter

if __name__ == '__main__':
    
    pose = uni_to_pose('kps3d/M1_01.uni')
    tc = np.array([-56.98, 129.79, -0.38])
    R, t = extrinsic_parameter(1.34, -1.47, -42.30, tc)
    p = mocap_to_camera(pose[0], R, t)
    fig = plt.figure()
    ax = Axes3D(fig)
    show_3d_pose(p, ax)
    fig.canvas.draw()
    plt.savefig('test.png')
    
