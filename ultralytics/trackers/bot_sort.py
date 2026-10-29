# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional, Union

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH, chi2inv95, chi2inv99, chi2inv999


class BOTrack(STrack):
    """An extended version of the STrack class for YOLO, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features: Update features vector and smooth it using exponential moving average.
        predict: Predict the mean and covariance using Kalman filter.
        re_activate: Reactivate a track with updated features and optionally new ID.
        update: Update the track with new detection and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict: Predict the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords: Convert tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh: Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(xywh=np.array([100, 50, 80, 40, 0]), score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(xywh=np.array([110, 60, 80, 40, 0]), score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(
        self, xywh: np.ndarray, score: float, cls: int, feat: Optional[np.ndarray] = None, feat_history: int = 50
    ):
        """Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is
                the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            feat (np.ndarray, optional): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.
        """
        super().__init__(xywh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque(maxlen=feat_history)
        self.alpha = 0.9
        self.q_nis_ema = 1.0

    def update_features(self, feat: np.ndarray) -> None:
        """Update the feature vector and apply exponential moving average smoothing."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self) -> None:
        """Predict the object's future state using the Kalman filter to update its mean and covariance."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None:
        """Reactivate a track with updated features and optionally assign a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track: BOTrack, frame_id: int) -> None:
        """Update the track with new detection information and the current frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self) -> np.ndarray:
        """Return the current bounding box position in `(top left x, top left y, width, height)` format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(
        stracks: list[BOTrack], q_scales: Optional[Union[np.ndarray, float]] = None, fade_lambda: float = 1.0
    ) -> None:
        """Predict the mean and covariance for multiple object tracks using a shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance, q_scales=q_scales, fade_lambda=fade_lambda
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: list[BOTrack], H: np.ndarray = np.eye(2, 3), fix_gmc_wh: bool = True) -> None:
        """Apply camera-motion compensation without rotating width/height state dimensions by default."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        R = H[:2, :2]
        t = H[:2, 2]
        if fix_gmc_wh:
            R8x8 = np.eye(8, dtype=float)
            R8x8[0:2, 0:2] = R
            R8x8[4:6, 4:6] = R
        else:
            R8x8 = np.kron(np.eye(4, dtype=float), R)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            mean = R8x8.dot(mean)
            mean[:2] += t
            cov = R8x8.dot(cov).dot(R8x8.transpose())
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert tlwh bounding box coordinates to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self) -> np.ndarray:
        """Return the current track box as an xywh Kalman measurement."""
        return self.tlwh_to_xywh(self.tlwh)


class BOTSORT(BYTETracker):
    """An extended version of the BYTETracker class for YOLO, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (Any): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter: Return an instance of KalmanFilterXYWH for object tracking.
        init_track: Initialize track with detection results and optional image for ReID.
        get_dists: Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict: Predict the mean and covariance of multiple object tracks using a shared Kalman filter.
        reset: Reset the BOTSORT tracker to its initial state.

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(results, img)
        >>> bot_sort.multi_predict(tracks)

    Notes:
        The class is designed to work with a YOLO object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args: Any, frame_rate: int = 30):
        """Initialize BOTSORT object with ReID module and GMC algorithm.

        Args:
            args (Any): Parsed command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video being processed.
        """
        super().__init__(args, frame_rate)
        self.gmc = GMC(method=args.gmc_method)

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = (
            (lambda feats, s: [f.cpu().numpy() for f in feats])  # native features do not require any model
            if args.with_reid and self.args.model == "auto"
            else ReID(args.model, getattr(args, "device", None))
            if args.with_reid
            else None
        )
        self._init_improved_botsort_options(args)

    def _init_improved_botsort_options(self, args: Any) -> None:
        """Load improved BoT-SORT options while preserving upstream defaults when options are absent."""
        self.enable_maha_gating = getattr(args, "enable_maha_gating", True)
        self.maha_only_position = getattr(args, "maha_only_position", True)
        self.maha_gating_prob = getattr(args, "maha_gating_prob", 0.999)
        self.maha_gating_tau = getattr(args, "maha_gating_tau", None)
        self.maha_inflate_alpha_pos = getattr(args, "maha_inflate_alpha_pos", 0.5)
        self.enable_maha_stage2 = getattr(args, "enable_maha_stage2", False)
        self.enable_maha_stage3 = getattr(args, "enable_maha_stage3", False)
        self.use_soft_motion = getattr(args, "use_soft_motion", True)
        self.debug_gating = getattr(args, "debug_gating", False)
        self.gating_log_interval = getattr(args, "gating_log_interval", 50)
        self.disable_hard_gate = getattr(args, "disable_hard_gate", True)
        self.min_std_xy = getattr(args, "min_std_xy", 5.0)
        self.fix_gmc_wh = getattr(args, "fix_gmc_wh", True)
        self.maha_tau_soft = getattr(args, "maha_tau_soft", None)

        gating_dim = 2 if self.maha_only_position else 4
        if self.maha_gating_tau is not None:
            self.tau_hard = self.maha_gating_tau
        elif self.maha_gating_prob >= 0.999:
            self.tau_hard = chi2inv999[gating_dim]
        elif self.maha_gating_prob >= 0.99:
            self.tau_hard = chi2inv99[gating_dim]
        else:
            self.tau_hard = chi2inv95[gating_dim]
        self.tau_soft = self.maha_tau_soft if self.maha_tau_soft is not None else chi2inv99[gating_dim]

        self.enable_adaptive_kf = getattr(args, "enable_adaptive_kf", False)
        self.akf_use_cam_motion = getattr(args, "akf_use_cam_motion", True)
        self.akf_q_cam_min = getattr(args, "akf_q_cam_min", 1.0)
        self.akf_q_cam_max = getattr(args, "akf_q_cam_max", 6.0)
        self.akf_rot_ref_deg = getattr(args, "akf_rot_ref_deg", 10.0)
        self.akf_k_rot = getattr(args, "akf_k_rot", 1.0)
        self.akf_k_trans = getattr(args, "akf_k_trans", 1.0)
        self.akf_k_scale = getattr(args, "akf_k_scale", 1.0)
        self.akf_use_nis = getattr(args, "akf_use_nis", True)
        self.akf_nis_dof = getattr(args, "akf_nis_dof", 2.0)
        self.akf_k_nis = getattr(args, "akf_k_nis", 0.5)
        self.akf_q_nis_max = getattr(args, "akf_q_nis_max", 4.0)
        self.akf_nis_ema = getattr(args, "akf_nis_ema", 0.9)
        self.akf_use_fading = getattr(args, "akf_use_fading", True)
        self.akf_lambda_min = getattr(args, "akf_lambda_min", 1.0)
        self.akf_lambda_max = getattr(args, "akf_lambda_max", 3.0)
        self.akf_debug = getattr(args, "akf_debug", False)
        self.akf_log_interval = getattr(args, "akf_log_interval", 50)

        self._q_cam = 1.0
        self._fade_lambda = 1.0

    def _compute_q_cam_from_warp(self, warp: np.ndarray, img_shape: tuple) -> tuple[float, dict]:
        """Compute an adaptive process-noise scale from GMC affine motion."""
        if not self.enable_adaptive_kf or not self.akf_use_cam_motion:
            return 1.0, {}
        R, t = warp[:2, :2], warp[:2, 2]
        rot_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        trans_px = math.sqrt(t[0] ** 2 + t[1] ** 2)
        s1 = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        s2 = math.sqrt(R[0, 1] ** 2 + R[1, 1] ** 2)
        scale_dev = 0.5 * (s1 + s2) - 1.0
        img_h, img_w = img_shape[:2]
        diag = max(math.sqrt(img_w**2 + img_h**2), 1.0)
        q_cam = 1.0 + (
            self.akf_k_rot * abs(rot_deg) / self.akf_rot_ref_deg
            + self.akf_k_trans * trans_px / diag
            + self.akf_k_scale * abs(scale_dev)
        )
        q_cam = max(self.akf_q_cam_min, min(self.akf_q_cam_max, q_cam))
        return q_cam, {"rot_deg": rot_deg, "trans_px": trans_px, "scale_dev": scale_dev}

    def _update_track_nis_ema(self, matches, strack_pool: list[BOTrack], detections: list[BOTrack]) -> None:
        """Update per-track adaptive process-noise EMA from matched innovation."""
        if not self.enable_adaptive_kf or not self.akf_use_nis:
            for track in strack_pool:
                track.q_nis_ema = 1.0
            return
        for itracked, idet in matches:
            track, det = strack_pool[itracked], detections[idet]
            if track.mean is None or track.covariance is None:
                continue
            measurement = det.to_xywh()[None, :]
            nis = self.kalman_filter.gating_distance(
                track.mean,
                track.covariance,
                measurement,
                only_position=self.maha_only_position,
                min_std=self.min_std_xy,
            )[0]
            q_nis = 1.0 + self.akf_k_nis * max(0.0, nis / self.akf_nis_dof - 1.0)
            q_nis = max(1.0, min(self.akf_q_nis_max, q_nis))
            track.q_nis_ema = self.akf_nis_ema * track.q_nis_ema + (1.0 - self.akf_nis_ema) * q_nis

    def _apply_maha_gating(
        self, dists: np.ndarray, tracks: list[BOTrack], detections: list[BOTrack], stage_name: str
    ) -> np.ndarray:
        """Apply the improved Mahalanobis gate/fusion used by the local BoT-SORT implementation."""
        if not self.enable_maha_gating or dists.size == 0:
            return dists
        if stage_name == "low_score" and not self.enable_maha_stage2:
            return dists
        if stage_name == "unconfirmed" and not self.enable_maha_stage3:
            return dists
        if any(t.mean is None or t.covariance is None for t in tracks):
            return dists

        need_stats = self.debug_gating and stage_name == "high_score" and self.frame_id % self.gating_log_interval == 0
        if self.use_soft_motion:
            result = matching.fuse_motion_soft(
                self.kalman_filter,
                dists,
                tracks,
                detections,
                only_position=self.maha_only_position,
                tau_hard=self.tau_hard,
                tau_soft=self.tau_soft,
                alpha_pos=self.maha_inflate_alpha_pos,
                disable_hard_gate=self.disable_hard_gate,
                min_std_xy=self.min_std_xy,
                return_stats=need_stats,
            )
            if need_stats:
                dists, stats = result
                LOGGER.info(
                    f"[BOTSORT-Maha] f={self.frame_id} tau={self.tau_soft:.1f}/{self.tau_hard:.1f} "
                    f"p50={stats.get('gating_dist_p50', 0):.2f} p90={stats.get('gating_dist_p90', 0):.2f}"
                )
            else:
                dists = result
        else:
            dists = matching.gate_cost_matrix(
                self.kalman_filter, dists, tracks, detections, self.maha_only_position, self.tau_hard
            )
        return dists

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """Return an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
        return KalmanFilterXYWH()

    def init_track(self, results, img: Optional[np.ndarray] = None) -> list[BOTrack]:
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder(img, bboxes)
            return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features_keep)]
        else:
            return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        """Calculate distances between tracks and detections using IoU and optionally ReID embeddings."""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def update(self, results, img: Optional[np.ndarray] = None, feats: Optional[np.ndarray] = None) -> np.ndarray:
        """Update tracks with improved BoT-SORT motion compensation, adaptive KF, and Mahalanobis gating."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]

        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results, feats_keep)
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        warp = np.eye(2, 3)
        if img is not None:
            try:
                warp = self.gmc.apply(img, results.xyxy)
            except Exception as e:
                LOGGER.warning(f"BoT-SORT GMC failed; using identity warp. {e}")

        self._q_cam, cam_stats = self._compute_q_cam_from_warp(warp, img.shape if img is not None else (1, 1))
        self._fade_lambda = (
            max(self.akf_lambda_min, min(self.akf_lambda_max, self._q_cam))
            if self.enable_adaptive_kf and self.akf_use_fading
            else 1.0
        )
        if not self.enable_adaptive_kf or not self.akf_use_nis:
            for st in strack_pool:
                st.q_nis_ema = 1.0

        self.multi_predict(strack_pool)
        BOTrack.multi_gmc(strack_pool, warp, fix_gmc_wh=self.fix_gmc_wh)
        BOTrack.multi_gmc(unconfirmed, warp, fix_gmc_wh=self.fix_gmc_wh)

        dists = self.get_dists(strack_pool, detections)
        dists = self._apply_maha_gating(dists, strack_pool, detections, "high_score")
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        self._update_track_nis_ema(matches, strack_pool, detections)

        if self.enable_adaptive_kf and self.akf_debug and self.frame_id % self.akf_log_interval == 0:
            q_nis = np.array([getattr(st, "q_nis_ema", 1.0) for st in strack_pool], dtype=float)
            LOGGER.info(
                f"[BOTSORT-AKF] f={self.frame_id} q_cam={self._q_cam:.2f} fade={self._fade_lambda:.2f} "
                f"rot={cam_stats.get('rot_deg', 0):.2f} trans={cam_stats.get('trans_px', 0):.1f} "
                f"q_nis_p50={np.percentile(q_nis, 50) if q_nis.size else 1.0:.2f}"
            )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        detections_second = self.init_track(results_second, feats_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections_second)
        dists = self._apply_maha_gating(dists, r_tracked_stracks, detections_second, "low_score")
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        dists = self._apply_maha_gating(dists, unconfirmed, detections, "unconfirmed")
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        """Predict the mean and covariance of multiple object tracks using a shared Kalman filter."""
        q_scales = None
        if self.enable_adaptive_kf and len(tracks) > 0:
            q_scales = np.array([self._q_cam * getattr(st, "q_nis_ema", 1.0) for st in tracks])
        BOTrack.multi_predict(tracks, q_scales=q_scales, fade_lambda=self._fade_lambda)

    def reset(self) -> None:
        """Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
        super().reset()
        self._q_cam = 1.0
        self._fade_lambda = 1.0
        self.gmc.reset_params()


class ReID:
    """YOLO model as encoder for re-identification."""

    def __init__(self, model: str, device: Optional[str] = None):
        """Initialize encoder for re-identification.

        Args:
            model (str): Path to the YOLO model for re-identification.
            device (str, optional): Device used by the ReID model, e.g. "cpu" or "0".
        """
        from ultralytics import YOLO

        self.model = YOLO(model)
        kwargs = {"embed": [len(self.model.model.model) - 2 if ".pt" in model else -1], "verbose": False, "save": False}
        if device:
            kwargs["device"] = device
        self.model(**kwargs)  # init

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for detected objects."""
        feats = self.model.predictor(
            [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        )
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().numpy() for f in feats]
