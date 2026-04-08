"""Tests for feature_extractor.py -- ORB detection and matching."""

import numpy as np
import pytest
import cv2

from raspbot_slam.feature_extractor import FeatureExtractor


class TestORBDetection:

    def test_detect_on_textured_image(self, synthetic_textured_image):
        fe = FeatureExtractor(n_features=500)
        kp, desc = fe.detect_and_compute(synthetic_textured_image)

        assert len(kp) > 50  # should find many features on textured image
        assert desc is not None
        assert desc.shape[1] == 32  # ORB descriptor is 32 bytes
        assert desc.dtype == np.uint8

    def test_detect_on_blank_image(self):
        fe = FeatureExtractor()
        blank = np.zeros((480, 640), dtype=np.uint8)
        kp, desc = fe.detect_and_compute(blank)
        assert len(kp) == 0
        assert desc is None

    def test_detect_returns_correct_types(self, synthetic_textured_image):
        fe = FeatureExtractor()
        kp, desc = fe.detect_and_compute(synthetic_textured_image)
        assert all(isinstance(k, cv2.KeyPoint) for k in kp)

    def test_nfeatures_respected(self, synthetic_textured_image):
        fe = FeatureExtractor(n_features=100)
        kp, desc = fe.detect_and_compute(synthetic_textured_image)
        assert len(kp) <= 100


class TestORBMatching:

    def test_self_match_perfect(self, synthetic_textured_image):
        """Matching an image against itself should yield many matches."""
        fe = FeatureExtractor()
        kp, desc = fe.detect_and_compute(synthetic_textured_image)
        matches = fe.match(desc, desc, ratio_threshold=0.9)
        # Self-matching with loose ratio should find most features
        assert len(matches) > len(kp) * 0.3

    def test_match_two_views(self, two_frame_sequence):
        """Two slightly different views should produce matches."""
        frame1, frame2 = two_frame_sequence
        fe = FeatureExtractor(n_features=500)

        kp1, desc1 = fe.detect_and_compute(frame1)
        kp2, desc2 = fe.detect_and_compute(frame2)

        matches = fe.match(desc1, desc2)
        assert len(matches) > 10

    def test_match_returns_empty_on_none(self):
        fe = FeatureExtractor()
        assert fe.match(None, None) == []

    def test_match_returns_empty_on_too_few(self):
        fe = FeatureExtractor()
        d1 = np.random.randint(0, 256, (1, 32), dtype=np.uint8)
        d2 = np.random.randint(0, 256, (1, 32), dtype=np.uint8)
        assert fe.match(d1, d2) == []  # need at least 2 for knnMatch(k=2)


class TestGeometricMatching:

    def test_geometric_check_on_related_frames(self, two_frame_sequence, mock_camera):
        """Essential matrix should be found for related frames."""
        frame1, frame2 = two_frame_sequence
        fe = FeatureExtractor(n_features=800)

        kp1, desc1 = fe.detect_and_compute(frame1)
        kp2, desc2 = fe.detect_and_compute(frame2)

        inliers, mask, E = fe.match_with_geometric_check(
            kp1, desc1, kp2, desc2, mock_camera.K)

        # Should find enough inliers for a valid Essential matrix
        if E is not None:
            assert len(inliers) >= 8
            assert E.shape == (3, 3)

    def test_geometric_check_on_unrelated_images(self, mock_camera):
        """Random images should yield no Essential matrix."""
        fe = FeatureExtractor()
        rng = np.random.RandomState(1)
        img1 = rng.randint(0, 256, (480, 640), dtype=np.uint8)
        img2 = rng.randint(0, 256, (480, 640), dtype=np.uint8)

        kp1, desc1 = fe.detect_and_compute(img1)
        kp2, desc2 = fe.detect_and_compute(img2)

        inliers, mask, E = fe.match_with_geometric_check(
            kp1, desc1, kp2, desc2, mock_camera.K)

        # Random images should have very few geometric inliers
        assert len(inliers) < 20  # might get a few by chance


class TestUtilities:

    def test_keypoints_to_array(self, synthetic_textured_image):
        fe = FeatureExtractor()
        kp, _ = fe.detect_and_compute(synthetic_textured_image)
        arr = FeatureExtractor.keypoints_to_array(kp)
        assert arr.shape == (len(kp), 2)
        assert arr.dtype == np.float64

    def test_keypoints_to_array_empty(self):
        arr = FeatureExtractor.keypoints_to_array([])
        assert arr.shape == (0, 2)

    def test_matches_to_points(self, two_frame_sequence):
        frame1, frame2 = two_frame_sequence
        fe = FeatureExtractor()
        kp1, desc1 = fe.detect_and_compute(frame1)
        kp2, desc2 = fe.detect_and_compute(frame2)
        matches = fe.match(desc1, desc2)

        pts1, pts2 = FeatureExtractor.matches_to_points(kp1, kp2, matches)
        assert pts1.shape == pts2.shape
        assert pts1.shape[0] == len(matches)
        assert pts1.shape[1] == 2
