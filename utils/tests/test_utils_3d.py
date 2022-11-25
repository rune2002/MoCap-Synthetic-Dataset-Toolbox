"""Test 3d keypoints related utils"""
import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from utils.data_utils import uni_to_pose, load_npz
from utils.scorer import Scorer

MOCAP_TO_HKMC =[25, 26, 17, 18, 19, 12, 13, 14, 5, 6, 1, 2]

def test_uni_to_pose():
    d = load_npz('./utils/tests/test_utils_3d/test_3d.npz')
    assert_array_equal(
        uni_to_pose('./utils/tests/test_utils_3d/M01_01.uni', MOCAP_TO_HKMC, downsample=3), d["M01_01"]
    )


def test_mpjpe():
    p1 = uni_to_pose('./utils/tests/test_utils_3d/M01_01.uni', MOCAP_TO_HKMC, downsample=3)
    p2 = np.copy(p1)
    assert Scorer.mpjpe(p1, p2) == 0


def test_mpjpe_byjoint():
    p1 = uni_to_pose('./utils/tests/test_utils_3d/M01_01.uni', MOCAP_TO_HKMC, downsample=3)
    p2 = np.copy(p1)
    assert_array_equal(
        Scorer.weighted_mpjpe(p1, p2, np.ones(len(MOCAP_TO_HKMC))), np.zeros(len(MOCAP_TO_HKMC))
    )


def test_weighted_mpjpe():
    p1 = uni_to_pose('./utils/tests/test_utils_3d/M01_01.uni', MOCAP_TO_HKMC, downsample=3)
    p2 = np.copy(p1)
    assert_array_equal(
        Scorer.mpjpe_byjoint(p1, p2), np.zeros(len(MOCAP_TO_HKMC))
    )


def test_pck():
    p1 = uni_to_pose('./utils/tests/test_utils_3d/M01_01.uni', MOCAP_TO_HKMC, downsample=3)
    p2 = np.copy(p1)
    assert Scorer.PCK(p1, p2, 150.0) == 100.0


def test_pck_byjoint():
    p1 = uni_to_pose('./utils/tests/test_utils_3d/M01_01.uni', MOCAP_TO_HKMC, downsample=3)
    p2 = np.copy(p1)
    assert_array_equal(
        Scorer.PCK_byjoint(p1, p2, 150.0), 100 * np.ones(len(MOCAP_TO_HKMC))
    )


def test_scorer():
    d = load_npz('./utils/tests/test_utils_3d/test_3d.npz')
    t = d.copy()
    s = Scorer(d, t, pck_threshold=150)
    assert s.score_all(s.mpjpe) == 0
    assert s.score_all(s.pck) == 100
    
    keys, scores = s.score_each(s.mpjpe)
    assert keys == ["M01_01"]
    assert scores == [0]

    keys, scores = s.score_each(s.pck)
    assert keys == ["M01_01"]
    assert scores == [100]