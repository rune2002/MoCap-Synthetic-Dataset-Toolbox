"""Utilities to deal with transformation between OptikTrack MoCap(let's say it is the world), and RGB camera coordinate"""

import math
import numpy as np

def mocap_to_camera(X_mocap, R, t):
    """Transform points from OptikTrack to RGB camera coordinates

    Args
        X_mocap: Nx3 3d points in world coordinates
        R: 3x3 camera extrinsic rotation matrix
        t: 3x1 camera extrinsic translation vector
    Return
        X_cam: Nx3 3d points in camera coordinates
    """
    
    X_cam = (R.dot(X_mocap.T) + t).T
    
    """Transform to conventional camera coordinates"""
    R_ = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    X_cam = R_.dot(X_cam.T).T 
    return X_cam


def camera_to_mocap(X_cam, R, t):
    """Transform points from RGB camera to OptikTrack coordinates

    Args
        X_cam: Nx3 3d points in conventional camera frame
        R: 3x3 camera extrinsic rotation matrix
        t: 3x1 camera extrinsic translation vector
    Return
        X_mocap: Nx3 3d points in world coordinates
    """

    """Transform to OptikTrack camera coordinates"""
    R_ = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    X_cam = R_.T.dot(X_cam.T).T 
    
    X_mocap = (R.T.dot(X_cam.T - t)).T
    return X_mocap

def extrinsic_parameter(x_deg, y_deg, z_deg, tvec):
    """Transform points from OptikTrack to RGB camera coordinates

    Args
        x_deg: x-axis rotation angle in degree
        y_deg: y-axis rotation angle in degree
        z_deg: z-axis rotation angle in degree
        tvec: 3x1 camera translation vector in world frame
    Return
        R: 3x3 camera extrinsic rotation matrix
        t: 3x1 camera extrinsic translation vector
    """
    R = euler_angle_to_rotation_matrix(x_deg, y_deg, z_deg).T
    t = - R.dot(tvec).reshape((3,1))
    return R, t

def euler_angle_to_rotation_matrix(x_deg, y_deg, z_deg):
    """Calculate rotation matrix from euler angle (xyz)

    Args
        x_deg: x-axis rotation angle in degree
        y_deg: y-axis rotation angle in degree
        z_deg: z-axis rotation angle in degree
    Return
        R: rotation matrix
    """
    x_rad = math.pi * (x_deg / 180)
    y_rad = math.pi * (y_deg / 180)
    z_rad = math.pi * (z_deg / 180)

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(x_rad), -math.sin(x_rad) ],
                    [0,         math.sin(x_rad), math.cos(x_rad)  ]
                    ])
 
    R_y = np.array([[math.cos(y_rad),    0,      math.sin(y_rad)  ],
                    [0,                     1,      0                   ],
                    [-math.sin(y_rad),   0,      math.cos(y_rad)  ]
                    ])
 
    R_z = np.array([[math.cos(z_rad),    -math.sin(z_rad),    0],
                    [math.sin(z_rad),    math.cos(z_rad),     0],
                    [0,                     0,                      1]
                    ])
 
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R