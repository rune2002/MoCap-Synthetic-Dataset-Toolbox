import os
import json
import imageio
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


class Viz:
    """Class for scoring 3D pose estimation"""
    def __init__(self, config):
        _, ext = os.path.splitext(config)
        assert ext == ".json"

        with open(config, 'r') as f:
            data = json.load(f)

        self.MOCAP_NAMES = data["MOCAP_NAMES"]
        self.HKMC_NAMES = data["HKMC_NAMES"]
        self.HKMC_ADJ_LIST = data["HKMC_ADJ_LIST"]


    def npz_to_images(self, d, output_dir):
        for key, value in d.items():
            self.poses_to_images(value, key, output_dir)


    def poses_to_images(self, poses, key, output_dir):
        fig = plt.figure()
        ax = Axes3D(fig)
        for i, pose in enumerate(poses):
            self.show_3d_pose(poses, ax)
            fig.canvas.draw()
            images_dir = os.path.join(output_dir, key)
            plt.savefig(os.path.join(images_dir, f'{i:03d}.png'))
            plt.clf()
        
        self.images_dir_to_gif(images_dir, output_dir)


    def show_3d_pose(self, pose, ax, add_labels=True):
        """Visualize a 3d skeleton"""
        
        assert np.shape(pose)[0] == len(self.HKMC_NAMES)
        # p = pose[:, :] - pose[:1, :]
        p = pose

        
        for i, j in self.HKMC_EDGES:
            ax.plot([p[i][0], p[j][0]], [p[i][1], p[j][1]], [p[i][2] , p[j][2]])
            # ax.plot([p[i][0], p[j][0]], [p[i][2], p[j][2]], [-p[i][1] , -p[j][1]])

        RADIUS = 100 # space around the subject

        # ax.set_xlim3d([-RADIUS, RADIUS])
        # ax.set_ylim3d([-RADIUS, RADIUS])
        # ax.set_zlim3d([-RADIUS, RADIUS])
        
        # ax.get_xaxis().set_ticklabels([])
        # ax.get_yaxis().set_ticklabels([])
        # ax.set_zticklabels([])
        # white = (1.0, 1.0, 1.0, 0.0)
        # ax.w_xaxis.set_pane_color(white)
        # ax.w_yaxis.set_pane_color(white)


    def images_dir_to_gif(self, images_dir, gif_path, fps=10):
        images = []
        for image in os.listdir(images_dir):
            images.append(imageio.imread(os.path.join(images_dir, image)))
        imageio.mimsave(gif_path, images, fps=fps)