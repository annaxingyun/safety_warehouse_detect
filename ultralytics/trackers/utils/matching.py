# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from typing import Optional, Union

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

from . import kalman_filter

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """Perform linear assignment using either the scipy or lap.lapjv method.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool): Use lap.lapjv for the assignment. If False, scipy.optimize.linear_sum_assignment is used.

    Returns:
        matched_indices (list[list[int]] | np.ndarray): Matched indices of shape (K, 2), where K is the number of
            matches.
        unmatched_a (tuple | list | np.ndarray): Unmatched indices from the first set.
        unmatched_b (tuple | list | np.ndarray): Unmatched indices from the second set.

    Examples:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Use scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).

    Examples:
        Compute IoU distance between two sets of tracks
        >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return 1 - ious  # cost matrix


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[BOTrack]): List of tracks, where each track contains embedding features.
        detections (list[BOTrack]): List of detections, where each detection contains embedding features.
        metric (str): Metric for distance computation. Supported metrics include 'cosine', 'euclidean', etc.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings with shape (N, M), where N is the number of tracks and M
            is the number of detections.

    Examples:
        Compute the embedding distance between tracks and detections using cosine metric
        >>> tracks = [BOTrack(...), BOTrack(...)]  # List of track objects with embedding features
        >>> detections = [BOTrack(...), BOTrack(...)]  # List of detection objects with embedding features
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def _track_measurements(detections: list) -> np.ndarray:
    """Return detections as xywh measurements for Kalman gating."""
    return np.asarray([det.xywh for det in detections], dtype=np.float32)


def gate_cost_matrix(
    kf,
    cost_matrix: np.ndarray,
    tracks: list,
    detections: list,
    only_position: bool = False,
    gating_threshold: Optional[float] = None,
) -> np.ndarray:
    """Apply Mahalanobis hard gating to a cost matrix."""
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    if gating_threshold is None:
        gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = _track_measurements(detections)
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion_soft(
    kf,
    cost_matrix: np.ndarray,
    tracks: list,
    detections: list,
    only_position: bool = True,
    lambda_: float = 0.98,
    tau_hard: float = 13.8155,
    tau_soft: float = 9.2103,
    alpha_pos: float = 0.5,
    normalize_motion: bool = True,
    disable_hard_gate: bool = False,
    min_std_xy: float = 0.0,
    return_stats: bool = False,
    return_gating_dist_matrix: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
    """Fuse association cost with a soft Mahalanobis motion penalty."""
    if cost_matrix.size == 0:
        return (cost_matrix, {}) if return_stats else cost_matrix

    measurements = _track_measurements(detections)
    gating_dist_matrix = np.full(cost_matrix.shape, np.inf, dtype=np.float64)

    for row, track in enumerate(tracks):
        if track.mean is None or track.covariance is None:
            continue
        cov = track.covariance.copy()
        if alpha_pos > 0:
            cov[0, 0] *= 1 + alpha_pos
            cov[1, 1] *= 1 + alpha_pos

        gating_dist = kf.gating_distance(
            track.mean, cov, measurements, only_position=only_position, metric="maha", min_std=min_std_xy
        )
        gating_dist_matrix[row] = gating_dist

        hard_mask = np.zeros(len(gating_dist), dtype=bool) if disable_hard_gate else gating_dist > tau_hard
        soft_mask = (gating_dist > tau_soft) & ~hard_mask
        if not disable_hard_gate:
            cost_matrix[row, hard_mask] = np.inf

        motion_cost = np.clip(gating_dist / tau_soft, 0, 3.0) if normalize_motion else gating_dist
        penalty = np.zeros_like(gating_dist)
        if tau_hard > tau_soft:
            penalty[soft_mask] = 0.1 * (gating_dist[soft_mask] - tau_soft) / (tau_hard - tau_soft)
        fused = lambda_ * cost_matrix[row] + (1 - lambda_) * motion_cost + penalty
        if not disable_hard_gate:
            fused[hard_mask] = np.inf
        cost_matrix[row] = fused

    if not return_stats:
        return cost_matrix

    finite = gating_dist_matrix[np.isfinite(gating_dist_matrix)]
    stats = (
        {
            "gating_dist_p50": float(np.percentile(finite, 50)),
            "gating_dist_p90": float(np.percentile(finite, 90)),
            "gating_dist_p99": float(np.percentile(finite, 99)),
            "gating_dist_min": float(np.min(finite)),
            "gating_dist_max": float(np.max(finite)),
        }
        if finite.size
        else {}
    )
    if return_gating_dist_matrix:
        stats["gating_dist_matrix"] = gating_dist_matrix
    return cost_matrix, stats


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse cost matrix with detection scores to produce a single cost matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (list[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (np.ndarray): Fused cost matrix with shape (N, M).

    Examples:
        Fuse a cost matrix with detection scores
        >>> cost_matrix = np.random.rand(5, 10)  # 5 tracks and 10 detections
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = det_scores[None].repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
