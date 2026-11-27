import collections
import json
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False


def _resolve_model_dir(preferred_dir: str = "") -> str:
    candidates = []
    if preferred_dir:
        candidates.append(preferred_dir)

    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.extend([
        os.path.join(pkg_root, "models"),
        "models",
    ])

    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("robosignformer")
        candidates.append(os.path.join(share_dir, "models"))
    except Exception:
        pass

    for d in candidates:
        if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, "robosignformer_config.json")):
            return d
    raise FileNotFoundError(
        "Model klasörü ('models/') bulunamadı! Lütfen 'robosignformer_jetson.onnx', "
        "'robosignformer_config.json' ve MediaPipe '.task' dosyalarının 'models/' dizininde olduğundan emin olun."
    )


class RoboSignFormerPipeline:
    """
    RoboSignFormer Gerçek Zamanlı Türk İşaret Dili Tanıma Boru Hattı (AUTSL 226 Sınıf).
    - 1.74M parametre, 108.20 MFLOPs, %91.37 tek-model (görülmemiş işaretçi) test doğruluğu.
    - PC (CPU/CUDA) ve NVIDIA Jetson Orin Nano (TensorRT/CUDA) üzerinde ONNXRuntime ile çalışır.
    """

    SLICE_POSE = slice(0, 8)
    SLICE_FACE = slice(8, 38)
    SLICE_LH   = slice(38, 59)
    SLICE_RH   = slice(59, 80)

    def __init__(
        self,
        model_dir: str = "",
        conf_threshold: float = 0.55,
        mirror_camera: bool = False,
        min_sign_frames: int = 10,
        max_sign_frames: int = 90,
        idle_end_frames: int = 6,
        live_preview_stride: int = 10,
    ):
        if not _HAS_ORT:
            raise ImportError("Lütfen 'pip install onnxruntime' (veya GPU için onnxruntime-gpu) kurun.")

        self.model_dir = _resolve_model_dir(model_dir)
        cfg_path = os.path.join(self.model_dir, "robosignformer_config.json")
        onnx_path = os.path.join(self.model_dir, "robosignformer_jetson.onnx")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.target_frames: int = int(self.cfg["target_frames"])
        self.total_lm: int = int(self.cfg["total_landmarks"])
        self.pose_indices: list[int] = list(self.cfg["pose_indices"])
        self.face_indices: list[int] = list(self.cfg["face_indices"])
        self.bones_52: list[list[int]] = list(self.cfg["bones_52"])
        self.class_map: dict[int, str] = {int(k): v for k, v in self.cfg["class_map"].items()}

        self.conf_threshold = float(conf_threshold)
        self.mirror_camera = bool(mirror_camera)
        self.min_sign_frames = int(min_sign_frames)
        self.max_sign_frames = int(max_sign_frames)
        self.idle_end_frames = int(idle_end_frames)
        self.live_preview_stride = int(live_preview_stride)

        available = ort.get_available_providers()
        preferred = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        # Windows Türkçe klasör yolu (Ü, ı, ö, ş, ç) uyumluluğu için byte dizisi üzerinden yükle
        with open(onnx_path, "rb") as f_onnx:
            onnx_bytes = f_onnx.read()
        self.ort_session = ort.InferenceSession(onnx_bytes, providers=preferred or ["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        print(f"🤖 [RoboSignFormer] ONNX aktif ({self.ort_session.get_providers()[0]}) -> {onnx_path}")

        pose_task = os.path.join(self.model_dir, "pose_landmarker.task")
        face_task = os.path.join(self.model_dir, "face_landmarker.task")
        hand_task = os.path.join(self.model_dir, "hand_landmarker.task")

        with open(pose_task, "rb") as fp, open(face_task, "rb") as ff, open(hand_task, "rb") as fh:
            pose_buf, face_buf, hand_buf = fp.read(), ff.read(), fh.read()

        self.pose_det = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=pose_buf),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        )
        self.face_det = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=face_buf),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )
        self.hand_det = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=hand_buf),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
        )

        self._ts_ms = 0
        self._pre_roll = collections.deque(maxlen=4)
        self._active_buffer: list[np.ndarray] = []
        self._state = "IDLE"
        self._idle_streak = 0
        self._active_streak = 0

        self.last_confirmed_word: str | None = None
        self.last_confirmed_score: float = 0.0
        self.last_top3: list[tuple[str, float]] = []
        self.live_preview_word: str | None = None
        self.live_preview_score: float = 0.0
        self._last_fps_t = time.perf_counter()
        self.fps_ema: float = 0.0

    def close(self):
        for det in (self.pose_det, self.face_det, self.hand_det):
            try:
                det.close()
            except Exception:
                pass

    @staticmethod
    def _assign_hands(pose_result, hand_result) -> tuple[np.ndarray, np.ndarray]:
        lh = np.zeros((21, 3), dtype=np.float32)
        rh = np.zeros((21, 3), dtype=np.float32)
        if not hand_result.hand_landmarks:
            return lh, rh

        hands_xyz = [
            np.array([[lm.x, lm.y, lm.z] for lm in h_lms], dtype=np.float32)
            for h_lms in hand_result.hand_landmarks[:2]
        ]

        if pose_result.pose_landmarks:
            p_lms = pose_result.pose_landmarks[0]
            lw = np.array([p_lms[15].x, p_lms[15].y], dtype=np.float32)
            rw = np.array([p_lms[16].x, p_lms[16].y], dtype=np.float32)
            if len(hands_xyz) == 1:
                hw = hands_xyz[0][0, :2]
                if np.linalg.norm(hw - lw) <= np.linalg.norm(hw - rw):
                    lh = hands_xyz[0]
                else:
                    rh = hands_xyz[0]
            else:
                w0, w1 = hands_xyz[0][0, :2], hands_xyz[1][0, :2]
                if (np.linalg.norm(w0 - lw) + np.linalg.norm(w1 - rw)) <= (np.linalg.norm(w0 - rw) + np.linalg.norm(w1 - lw)):
                    lh, rh = hands_xyz[0], hands_xyz[1]
                else:
                    rh, lh = hands_xyz[0], hands_xyz[1]
        else:
            assigned_l, assigned_r = False, False
            for i, h_arr in enumerate(hands_xyz):
                label = hand_result.handedness[i][0].category_name
                if label == "Left" and not assigned_l:
                    lh, assigned_l = h_arr, True
                elif label == "Right" and not assigned_r:
                    rh, assigned_r = h_arr, True
                elif not assigned_l:
                    lh, assigned_l = h_arr, True
                elif not assigned_r:
                    rh, assigned_r = h_arr, True
        return lh, rh

    @staticmethod
    def _interpolate_group(seq_group: np.ndarray, max_gap: int = 5, fill_edges: bool = False) -> np.ndarray:
        T = seq_group.shape[0]
        valid = np.any(np.abs(seq_group) > 1e-4, axis=(1, 2))
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return seq_group

        out = seq_group.copy()
        for i in range(len(valid_idx) - 1):
            s, e = valid_idx[i], valid_idx[i + 1]
            gap = e - s - 1
            if 0 < gap <= max_gap:
                for t in range(s + 1, e):
                    alpha = (t - s) / float(e - s)
                    out[t] = (1.0 - alpha) * out[s] + alpha * out[e]

        if fill_edges:
            first_v, last_v = valid_idx[0], valid_idx[-1]
            if first_v > 0:
                out[:first_v] = out[first_v]
            if last_v < T - 1:
                out[last_v + 1:] = out[last_v]
        return out

    def _prepare_32_frames(self, raw_frames: list[np.ndarray]) -> np.ndarray:
        seg = np.stack(raw_frames, axis=0).astype(np.float32)
        seg_len = seg.shape[0]
        if seg_len >= self.target_frames:
            idxs = np.linspace(0, seg_len - 1, self.target_frames, dtype=int)
        else:
            idxs = np.concatenate([
                np.arange(seg_len, dtype=int),
                np.full(self.target_frames - seg_len, seg_len - 1, dtype=int),
            ])
        seq = seg[idxs].copy()
        seq[:, self.SLICE_POSE] = self._interpolate_group(seq[:, self.SLICE_POSE], max_gap=self.target_frames, fill_edges=True)
        seq[:, self.SLICE_FACE] = self._interpolate_group(seq[:, self.SLICE_FACE], max_gap=self.target_frames, fill_edges=True)
        seq[:, self.SLICE_LH]   = self._interpolate_group(seq[:, self.SLICE_LH],   max_gap=5, fill_edges=False)
        seq[:, self.SLICE_RH]   = self._interpolate_group(seq[:, self.SLICE_RH],   max_gap=5, fill_edges=False)
        return seq

    @staticmethod
    def _mirror_seq80(seq_32: np.ndarray) -> np.ndarray:
        """Anatomik sol/sağ el değişmezliği (handedness invariance) için 80 eklemi yatay yansıtır."""
        s = seq_32.copy()
        valid = np.any(np.abs(s) > 1e-4, axis=-1)
        s[valid, 0] = 1.0 - s[valid, 0]
        s[:, [0, 1, 2, 3, 4, 5, 6, 7]] = s[:, [1, 0, 3, 2, 5, 4, 7, 6]]
        f_even = s[:, 18:38:2].copy()
        s[:, 18:38:2] = s[:, 19:38:2]
        s[:, 19:38:2] = f_even
        lh_cp = s[:, 38:59].copy()
        s[:, 38:59] = s[:, 59:80]
        s[:, 59:80] = lh_cp
        return s

    @staticmethod
    def _softmax_1d(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        return exp_l / np.sum(exp_l)

    def predict_sequence(self, raw_frames: list[np.ndarray]) -> list[tuple[str, float]]:
        seq_32 = self._prepare_32_frames(raw_frames)
        inp_raw = seq_32[np.newaxis, ...].astype(np.float32)
        logits_raw = self.ort_session.run([self.output_name], {self.input_name: inp_raw})[0][0]
        probs_raw = self._softmax_1d(logits_raw)

        seq_mir = self._mirror_seq80(seq_32)
        inp_mir = seq_mir[np.newaxis, ...].astype(np.float32)
        logits_mir = self.ort_session.run([self.output_name], {self.input_name: inp_mir})[0][0]
        probs_mir = self._softmax_1d(logits_mir)

        probs = probs_raw if float(np.max(probs_raw)) >= float(np.max(probs_mir)) else probs_mir
        top_idx = np.argsort(probs)[::-1][:3]
        return [(self.class_map.get(int(i), str(i)), float(probs[i])) for i in top_idx]

    @staticmethod
    def _is_hand_signing_active(pts_80: np.ndarray) -> bool:
        lh_ok = bool(np.any(np.abs(pts_80[38:59]) > 1e-4))
        rh_ok = bool(np.any(np.abs(pts_80[59:80]) > 1e-4))
        if not (lh_ok or rh_ok):
            return False

        sh_ok = (abs(pts_80[0, 1]) > 1e-4) and (abs(pts_80[1, 1]) > 1e-4)
        hips_ok = (abs(pts_80[6, 1]) > 1e-4) and (abs(pts_80[7, 1]) > 1e-4)
        thresh_y = 0.92
        if hips_ok and sh_ok:
            sh_y = 0.5 * (pts_80[0, 1] + pts_80[1, 1])
            hip_y = 0.5 * (pts_80[6, 1] + pts_80[7, 1])
            thresh_y = min(0.92, max(hip_y - 0.02, sh_y + 0.34))
        elif sh_ok:
            sh_y = 0.5 * (pts_80[0, 1] + pts_80[1, 1])
            thresh_y = min(0.92, sh_y + 0.38)
        lw_above = lh_ok and (pts_80[38, 1] < thresh_y)
        rw_above = rh_ok and (pts_80[59, 1] < thresh_y)
        return bool(lw_above or rw_above)

    def _draw_hud(self, bgr_frame: np.ndarray, pts_80: np.ndarray) -> np.ndarray:
        vis = bgr_frame.copy()
        h, w, _ = vis.shape

        for p_i, c_i in self.bones_52:
            if (abs(pts_80[p_i, 0]) + abs(pts_80[p_i, 1]) > 1e-4) and (abs(pts_80[c_i, 0]) + abs(pts_80[c_i, 1]) > 1e-4):
                cv2.line(
                    vis,
                    (int(pts_80[p_i, 0] * w), int(pts_80[p_i, 1] * h)),
                    (int(pts_80[c_i, 0] * w), int(pts_80[c_i, 1] * h)),
                    (180, 180, 180), 1,
                )

        for sl, col, rad in (
            (self.SLICE_POSE, (60, 60, 255), 4),
            (self.SLICE_FACE, (0, 220, 255), 2),
            (self.SLICE_LH,   (255, 200, 0), 3),
            (self.SLICE_RH,   (80, 255, 50), 3),
        ):
            for px, py, _ in pts_80[sl]:
                if abs(px) + abs(py) > 1e-4:
                    cv2.circle(vis, (int(px * w), int(py * h)), rad, col, -1)

        cv2.rectangle(vis, (0, 0), (w, 82), (25, 25, 25), -1)
        state_col = (0, 220, 255) if self._state == "SIGNING" else (180, 180, 180)
        status_txt = (
            f"Durum: ISARET IZLENIYOR ({len(self._active_buffer)} kare)"
            if self._state == "SIGNING"
            else "Durum: HAZIR (Isaret icin elinizi kaldirin)"
        )
        cv2.putText(vis, f"{status_txt} | FPS: {self.fps_ema:.1f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, state_col, 2)

        if self._state == "SIGNING" and self.live_preview_word:
            cv2.putText(
                vis,
                f"Canli Onizleme: {self.live_preview_word} ({self.live_preview_score*100:.0f}%)",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 210, 90), 2,
            )
        elif self.last_confirmed_word:
            top3_str = " | ".join(f"{wd} ({sc*100:.0f}%)" for wd, sc in self.last_top3)
            cv2.putText(
                vis,
                f"Son Tahmin: {self.last_confirmed_word.upper()} ({self.last_confirmed_score*100:.1f}%)  [{top3_str}]",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 255, 100), 2,
            )
        return vis

    def process_frame(self, frame_bgr: np.ndarray, draw_hud: bool = True) -> tuple[str | None, int, np.ndarray]:
        now = time.perf_counter()
        dt = max(1e-4, now - self._last_fps_t)
        self._last_fps_t = now
        inst_fps = 1.0 / dt
        self.fps_ema = inst_fps if self.fps_ema == 0.0 else (0.9 * self.fps_ema + 0.1 * inst_fps)

        work_bgr = cv2.flip(frame_bgr, 1) if self.mirror_camera else frame_bgr
        rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        step_ms = max(1, int(round(dt * 1000.0)))
        self._ts_ms += step_ms

        p_res = self.pose_det.detect_for_video(mp_img, self._ts_ms)
        f_res = self.face_det.detect_for_video(mp_img, self._ts_ms)
        h_res = self.hand_det.detect_for_video(mp_img, self._ts_ms)

        pts = np.zeros((self.total_lm, 3), dtype=np.float32)
        if p_res.pose_landmarks:
            p_lms = p_res.pose_landmarks[0]
            for i, idx in enumerate(self.pose_indices):
                lm = p_lms[idx]
                pts[i] = [lm.x, lm.y, lm.z]
        if f_res.face_landmarks:
            f_lms = f_res.face_landmarks[0]
            for i, idx in enumerate(self.face_indices):
                lm = f_lms[idx]
                pts[8 + i] = [lm.x, lm.y, lm.z]
        lh, rh = self._assign_hands(p_res, h_res)
        pts[self.SLICE_LH] = lh
        pts[self.SLICE_RH] = rh

        active_now = self._is_hand_signing_active(pts)
        confirmed_word: str | None = None
        confirmed_score_int: int = 0

        if self._state == "IDLE":
            self._pre_roll.append(pts)
            if active_now:
                self._active_streak += 1
                if self._active_streak >= 3:
                    self._state = "SIGNING"
                    self._active_buffer = list(self._pre_roll)
                    self._idle_streak = 0
                    self.live_preview_word = None
            else:
                self._active_streak = 0

        elif self._state == "SIGNING":
            self._active_buffer.append(pts)
            if active_now:
                self._idle_streak = 0
            else:
                self._idle_streak += 1

            n_buf = len(self._active_buffer)
            if n_buf >= 18 and (n_buf % self.live_preview_stride == 0):
                top3_live = self.predict_sequence(self._active_buffer)
                self.live_preview_word, self.live_preview_score = top3_live[0]

            if self._idle_streak >= self.idle_end_frames or n_buf >= self.max_sign_frames:
                trim_end = max(self.min_sign_frames, n_buf - max(0, self._idle_streak - 2))
                gesture_clip = self._active_buffer[:trim_end]

                if len(gesture_clip) >= self.min_sign_frames:
                    top3 = self.predict_sequence(gesture_clip)
                    best_w, best_p = top3[0]
                    if best_p >= self.conf_threshold:
                        self.last_confirmed_word = best_w
                        self.last_confirmed_score = best_p
                        self.last_top3 = top3
                        confirmed_word = best_w
                        confirmed_score_int = int(round(best_p * 100.0))

                self._state = "IDLE"
                self._active_buffer.clear()
                self._pre_roll.clear()
                self._idle_streak = 0
                self._active_streak = 0
                self.live_preview_word = None

        vis_frame = self._draw_hud(work_bgr, pts) if draw_hud else work_bgr
        return confirmed_word, confirmed_score_int, vis_frame
