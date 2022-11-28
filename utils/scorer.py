"""
This module scores 3D pose estimation
"""

import os
import numpy as np
import numpy.ma as ma

class Scorer:
    """Class for scoring 3D pose estimation"""

    def __init__(self, submission, target, pck_threshold=150):
        self.d = submission
        self.t = target
        self.pck_threshold = pck_threshold

    def report(self):
        mpjpe_all = self.score_all(self.mpjpe)
        tests, mpjpe_each = self.score_each(self.mpjpe)
        pck_all = self.score_all(self.pck)
        tests, PCK_each = self.score_each(self.pck)
        print(mpjpe_all)
        print(pck_all)

    def score_all(self, metric):
        predicted = None
        target = None
        for key in list(self.d.keys()):
            assert key in list(self.t.keys())
            if predicted:
                predicted = np.append(predicted, self.d[key], axis=0)
            else:
                predicted = np.array(self.d[key])

            if target:
                target = np.append(target, self.t[key], axis=0)
            else:
                target = np.array(self.t[key])

        return metric(predicted, target)

    def score_each(self, metric):
        scores = []
        for key in list(self.d.keys()):
            assert key in list(self.t.keys())
            scores.append(metric(self.d[key], self.t[key]))
        return list(self.d.keys()), scores

    @staticmethod
    def mpjpe(predicted, target):
        """Mean per-joint position error (i.e. mean Euclidean distance)"""
        assert predicted.shape == target.shape
        mask = Scorer.valid_mask(predicted)
        p_ma = ma.masked_array(predicted, mask)
        t_ma = ma.masked_array(target, mask)
        return np.mean(np.linalg.norm(p_ma - t_ma, axis=len(t_ma.shape) - 1))

    @staticmethod
    def mpjpe_byjoint(predicted, target):
        """Mean per-joint position error (i.e. mean Euclidean distance)"""
        assert predicted.shape == target.shape
        mask = Scorer.valid_mask(predicted)
        p_ma = ma.masked_array(predicted, mask)
        t_ma = ma.masked_array(target, mask)
        return np.mean(np.linalg.norm(p_ma - t_ma, axis=len(t_ma.shape) - 1), axis=0)

    @staticmethod
    def weighted_mpjpe(predicted, target, w):
        """Weighted mean per-joint position error (i.e. mean Euclidean distance)"""
        assert predicted.shape == target.shape
        assert w.shape[0] == predicted.shape[1]
        mask = Scorer.valid_mask(predicted)
        p_ma = ma.masked_array(predicted, mask)
        t_ma = ma.masked_array(target, mask)
        return np.mean(w * np.linalg.norm(p_ma - t_ma, axis=len(t_ma.shape) - 1))

    def pck(self, predicted, target):
        return self.PCK(predicted, target, self.pck_threshold)

    def pck_byjoint(self, predicted, target):
        return self.PCK_byjoint(predicted, target, self.pck_threshold)

    @staticmethod
    def PCK(predicted, target, PCK_THRESHOLD):
        sample_num = len(target)
        total = 0
        true_positive = 0
        mask = Scorer.valid_mask(predicted).all(axis=-1)

        for n in range(sample_num):
            gt = target[n]
            pred = predicted[n]
            per_joint_error = np.sqrt(np.sum(np.power(pred - gt, 2), 1))
            true_positive += np.minimum(per_joint_error < PCK_THRESHOLD, ~mask[n]).sum()
            total += per_joint_error.size

        pck = float(true_positive / total) * 100
        return pck

    @staticmethod
    def PCK_byjoint(predicted, target, PCK_THRESHOLD):
        sample_num = len(target)
        total = 0
        true_positive = np.zeros(target.shape[1])
        mask = Scorer.valid_mask(predicted).all(axis=-1)

        for n in range(sample_num):
            gt = target[n]
            pred = predicted[n]
            per_joint_error = np.sqrt(np.sum(np.power(pred - gt, 2), 1))
            true_positive += np.minimum(per_joint_error < PCK_THRESHOLD, ~mask[n])
            total += 1

        pck = true_positive / total * 100
        return pck
    
    @staticmethod
    def valid_mask(pose):
        assert pose.shape[-1] == 3
        v = np.expand_dims((pose == np.zeros_like(pose)).all(axis=-1), axis=-1)
        return np.concatenate((v, v, v), axis=-1)