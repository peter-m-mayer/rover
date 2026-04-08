"""
ORB feature detection, description, and matching.

Provides the visual feature pipeline used by both visual odometry and
synthetic stereo. ORB is chosen for Pi 5 real-time performance (~20ms/frame
at 640x480 with 1000 features, vs ~200ms for SIFT).
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import config


class FeatureExtractor:
    """ORB-based feature detection and matching with outlier rejection."""

    def __init__(self, n_features=None, scale_factor=None, n_levels=None):
        if cv2 is None:
            raise RuntimeError("OpenCV required for FeatureExtractor")

        self._orb = cv2.ORB_create(
            nfeatures=n_features or config.ORB_N_FEATURES,
            scaleFactor=scale_factor or config.ORB_SCALE_FACTOR,
            nLevels=n_levels or config.ORB_N_LEVELS,
            edgeThreshold=config.ORB_EDGE_THRESHOLD,
            patchSize=config.ORB_PATCH_SIZE,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, frame: np.ndarray
                           ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extract ORB keypoints and descriptors from a grayscale frame.

        Args:
            frame: HxW uint8 grayscale image.

        Returns:
            (keypoints, descriptors) where descriptors is Nx32 uint8,
            or ([], None) if no features found.
        """
        kp, desc = self._orb.detectAndCompute(frame, None)
        if kp is None or len(kp) == 0:
            return [], None
        return kp, desc

    def match(self, desc1: np.ndarray, desc2: np.ndarray,
              ratio_threshold: float = None) -> List[cv2.DMatch]:
        """Match descriptors using BFMatcher with Lowe's ratio test.

        Args:
            desc1: Nx32 descriptors from frame 1.
            desc2: Mx32 descriptors from frame 2.
            ratio_threshold: Lowe's ratio. Defaults to config.LOWE_RATIO_THRESHOLD.

        Returns:
            List of good DMatch objects passing the ratio test.
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        ratio = ratio_threshold or config.LOWE_RATIO_THRESHOLD
        matches = self._matcher.knnMatch(desc1, desc2, k=2)

        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)
        return good

    def match_with_geometric_check(
        self,
        kp1: List[cv2.KeyPoint], desc1: np.ndarray,
        kp2: List[cv2.KeyPoint], desc2: np.ndarray,
        K: np.ndarray,
        ratio_threshold: float = None,
    ) -> Tuple[List[cv2.DMatch], np.ndarray, Optional[np.ndarray]]:
        """Match features with Lowe's ratio test + RANSAC Essential matrix.

        Three-layer outlier rejection:
        1. Lowe's ratio test (~60% false match rejection)
        2. RANSAC on Essential matrix (geometric consistency)
        3. Chirality check in recoverPose (rejects behind-camera points)

        Args:
            kp1, desc1: Keypoints and descriptors from frame 1.
            kp2, desc2: Keypoints and descriptors from frame 2.
            K: 3x3 camera intrinsic matrix.
            ratio_threshold: Lowe's ratio. Defaults to config.

        Returns:
            (inlier_matches, inlier_mask, E) where:
            - inlier_matches: DMatch objects that passed all checks
            - inlier_mask: boolean mask over the ratio-test matches
            - E: 3x3 Essential matrix, or None if too few inliers
        """
        # Layer 1: Lowe's ratio test
        good_matches = self.match(desc1, desc2, ratio_threshold)

        if len(good_matches) < config.MIN_INLIER_MATCHES:
            return [], np.array([]), None

        # Extract matched point coordinates
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float64)

        # Layer 2: RANSAC Essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=config.RANSAC_PROB,
            threshold=config.RANSAC_THRESHOLD,
        )

        if E is None or mask is None:
            return [], np.array([]), None

        mask = mask.ravel().astype(bool)
        inlier_matches = [m for m, is_inlier in zip(good_matches, mask) if is_inlier]

        if len(inlier_matches) < config.MIN_INLIER_MATCHES:
            return [], mask, None

        return inlier_matches, mask, E

    @staticmethod
    def keypoints_to_array(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """Convert keypoint list to Nx2 numpy array of (x, y) coordinates."""
        if not keypoints:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([kp.pt for kp in keypoints], dtype=np.float64)

    @staticmethod
    def matches_to_points(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                          matches: List[cv2.DMatch]
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract matched point pairs from keypoints and matches.

        Returns:
            (pts1, pts2) each Nx2 arrays of corresponding (x, y) coordinates.
        """
        if not matches:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)
        return pts1, pts2
