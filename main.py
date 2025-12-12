"""
Full working detector with:
 - YOLO (yolov8s / yolov8s-pose)
 - FER emotion detection
 - SlowFast R50 pretrained via PyTorchVideo (torch.hub)
 - Alerts + beep **ONLY** on fall / unsafe pose (other detections are still annotated)

Install prerequisites (run once):
    pip install ultralytics opencv-python-headless==4.7.0.72 cvzone fer imageio imageio-ffmpeg
    pip install torch torchvision  # pick the correct version matching your CUDA
    pip install pytorchvideo

If you prefer CPU-only and no GPU, predictions will be very slow.
"""
import os
import time
import math
from collections import deque, defaultdict
import json
import urllib.request
import platform

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from fer import FER

import torch
import torch.nn.functional as F

# -------------------------
# Configure models & paths
# -------------------------
POSE_MODEL_PATH = "yolov8s-pose.pt"
MODEL_PATH = 'yolov8s.pt'
CLASSES_PATH = 'classes.txt'

# SlowFast config
SLOWFAST_CLIP_LEN = 32          # number of frames per clip (model temporal context)
SLOWFAST_STRIDE = 16            # run inference every N frames (sliding window stride)
SLOWFAST_CROP = 224             # spatial crop size passed to model

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Basic checks
# -------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Classes file not found: {CLASSES_PATH}")
if not os.path.exists(POSE_MODEL_PATH):
    raise FileNotFoundError(f"Pose model file not found: {POSE_MODEL_PATH}")

# -------------------------
# Load YOLO models
# -------------------------
model = YOLO(MODEL_PATH)
pose_model = YOLO(POSE_MODEL_PATH)

classnames = []
with open(CLASSES_PATH, 'r') as f:
    classnames = f.read().splitlines()

# -------------------------
# Load FER emotion detector
# -------------------------
# keep original variable in case other code used it
emotion_detector = None
try:
    emotion_detector = FER(mtcnn=True)
except Exception:
    emotion_detector = None

# We'll also use a small wrapper for consistent API
class EmotionDetector:
    def __init__(self):
        try:
            self.detector = FER(mtcnn=True)
        except Exception:
            self.detector = None

    def detect_emotion(self, frame):
        if self.detector is None:
            return None
        try:
            emotions = self.detector.detect_emotions(frame)
            if emotions:
                top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
                return top_emotion
        except Exception:
            pass
        return None

emotion_detector_wrapper = EmotionDetector()

# -------------------------
# Load SlowFast (TorchHub / PyTorchVideo)
# -------------------------
print("Loading SlowFast model (this may download weights the first time)...")
try:
    slowfast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    slowfast_model = slowfast_model.eval().to(DEVICE)
    print("SlowFast loaded on", DEVICE)
except Exception as e:
    slowfast_model = None
    print("Warning: SlowFast model failed to load:", e)
    print("You can remove SlowFast calls or install pytorchvideo and try again.")

# Load Kinetics id->label map from HF (best-effort)
KINETICS_LABELS = None
try:
    hf_url = "https://raw.githubusercontent.com/huggingface/datasets/main/datasets/label-files/kinetics400-id2label.json"
    with urllib.request.urlopen(hf_url, timeout=10) as resp:
        id2label = json.load(resp)
    KINETICS_LABELS = {int(k): v for k, v in id2label.items()}
except Exception:
    KINETICS_LABELS = None

# Aggression keyword set (map kinetics label strings to aggression)
AGGR_KEYWORDS = ("punch", "fight", "boxing", "kick", "slap", "assault", "hit", "attack", "fighting")

# -------------------------
# Helper: SlowFast preprocessing & inference
# -------------------------
def frames_to_slowfast_tensor(frames, clip_len=SLOWFAST_CLIP_LEN, crop_size=SLOWFAST_CROP):
    """
    frames: list of RGB uint8 arrays (H,W,3)
    returns: [1, 3, T, H, W] tensor ready for the model
    """
    # ensure enough frames by repeating last frame if needed
    if len(frames) < clip_len:
        frames = frames + [frames[-1]] * (clip_len - len(frames))
    else:
        # uniformly sample clip_len frames across the buffer
        idxs = np.linspace(0, len(frames)-1, clip_len).astype(int)
        frames = [frames[i] for i in idxs]

    processed = []
    for f in frames:
        if f.dtype != np.uint8:
            f = (f * 255).astype('uint8')
        h, w = f.shape[:2]
        # resize shorter side to 256 (common preprocessing)
        if h < w:
            new_h = 256
            new_w = int(w * 256 / h)
        else:
            new_w = 256
            new_h = int(h * 256 / w)
        resized = cv2.resize(f, (new_w, new_h))
        # center crop
        startx = (new_w - crop_size) // 2
        starty = (new_h - crop_size) // 2
        cropped = resized[starty:starty+crop_size, startx:startx+crop_size]
        processed.append(cropped.astype(np.float32) / 255.0)

    arr = np.stack(processed, axis=0)  # (T,H,W,3)
    # normalize with simple mean/std (used in many video models)
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    std  = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # transpose to (C, T, H, W)
    arr = np.transpose(arr, (3, 0, 1, 2))
    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)  # [1, C, T, H, W]
    return tensor

def slowfast_predict(frames, topk=3):
    """
    frames: list of RGB frames (numpy arrays)
    returns: list of (label, score) topk predictions OR [] on failure
    """
    if slowfast_model is None:
        return []

    try:
        inp = frames_to_slowfast_tensor(frames)
        with torch.no_grad():
            logits = slowfast_model(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            topk_idxs = probs.argsort()[-topk:][::-1]
            results = []
            for idx in topk_idxs:
                label = KINETICS_LABELS.get(int(idx), str(idx)) if KINETICS_LABELS else str(idx)
                results.append((label, float(probs[idx])))
            return results
    except Exception as e:
        # on any error, return empty
        print("SlowFast inference error:", e)
        return []

# -------------------------
# small cross-platform beep helper
# -------------------------
def play_beep(freq=1000, length_ms=250):
    """
    Try to play a short beep. Uses winsound on Windows.
    Falls back to printing a bell character which may work on some terminals.
    """
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(freq, length_ms)
        else:
            # Unix: try beep via system bell
            print("\a", end="", flush=True)
    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass

# -------------------------
# ActionDetector (simplified, with per-person arm smoothing & improved hip history)
# -------------------------
class ActionDetector:
    def __init__(self, history_length=10):
        # hip histories increased length for more robust walking detection
        self.right_hip_history = defaultdict(lambda: deque(maxlen=history_length))
        self.left_hip_history = defaultdict(lambda: deque(maxlen=history_length))
        # per-person arm angle history to avoid mixing people
        self.prev_arm_angles = defaultdict(lambda: deque(maxlen=3))  # stores (left_angle, right_angle)
        self.last_move_time = defaultdict(lambda: 0.0)
        self.min_confidence = 0.25  # threshold to skip low-confidence pose estimates

    def angle(self, a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c
        ang = math.degrees(
            math.atan2(cy - by, cx - bx) -
            math.atan2(ay - by, ax - bx)
        )
        return abs(ang)

    def dist(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _avg_angle(self, pid):
        arr = self.prev_arm_angles[pid]
        if not arr:
            return None, None
        avg_left = sum(a[0] for a in arr) / len(arr)
        avg_right = sum(a[1] for a in arr) / len(arr)
        return avg_left, avg_right

    def detect_action(self, keypoints):
        """
        keypoints: list of detected person keypoints from yolov8 pose output.
        We expect kp.xy[0] to contain [x,y] or [x,y,conf] per point as before.
        Returns first strong action found (keeps your original API).
        """
        action = None

        for pid, kp in enumerate(keypoints):
            # read raw points and possibly confidences
            try:
                pts_raw = kp.xy[0].tolist()
            except Exception:
                # fallback if structure different
                try:
                    pts_raw = list(kp)
                except Exception:
                    continue

            # determine if confidence values are present
            has_conf = isinstance(pts_raw[0], (list, tuple)) and len(pts_raw[0]) >= 3
            # compute average confidence and skip if too low
            avg_conf = 1.0
            if has_conf:
                confs = [p[2] for p in pts_raw]
                avg_conf = sum(confs) / len(confs)
                if avg_conf < self.min_confidence:
                    # unreliable pose -> skip this person
                    continue

            # normalize pts list to simple (x,y) tuples
            pts = []
            for p in pts_raw:
                if isinstance(p, (list, tuple)):
                    pts.append((float(p[0]), float(p[1])))
                else:
                    # fallback
                    pts.append((0.0, 0.0))

            # map indices (consistent with your previous code)
            # safe indexing in case pose has fewer points
            def idx(i):
                return pts[i] if i < len(pts) else (0.0, 0.0)

            nose = idx(0)
            left_sh = idx(5)
            right_sh = idx(6)
            left_elb = idx(7)
            right_elb = idx(8)
            left_wri = idx(9)
            right_wri = idx(10)
            left_hip = idx(11)
            right_hip = idx(12)
            left_knee = idx(13)
            right_knee = idx(14)

            # --------------------
            # BASIC ACTIONS (simple)
            # --------------------
            # DRINKING (hand-to-mouth proximity)
            if self.dist(left_wri, nose) < 50 or self.dist(right_wri, nose) < 50:
                return "Drinking"

            # HANDS UP
            wrist_y = (left_wri[1] + right_wri[1]) / 2
            shoulder_y = (left_sh[1] + right_sh[1]) / 2
            if wrist_y < shoulder_y - 20:
                return "Hands Raised"

            # STANDING / SITTING
            hip_y = (left_hip[1] + right_hip[1]) / 2
            if hip_y < nose[1] + 50:
                action = "Standing"
            else:
                action = "Sitting"

            # WALKING (hip movement) -- use per-person hip histories to avoid mixing
            self.right_hip_history[pid].append(right_hip[0])
            self.left_hip_history[pid].append(left_hip[0])

            if len(self.right_hip_history[pid]) >= self.right_hip_history[pid].maxlen:
                delta_R = max(self.right_hip_history[pid]) - min(self.right_hip_history[pid])
                delta_L = max(self.left_hip_history[pid]) - min(self.left_hip_history[pid])
                # slightly lower threshold in normalized/resized frame coordinates
                if delta_R > 15 or delta_L > 15:
                    action = "Walking"

            # --------------------
            # AGGRESSION (replace global prev angles with per-person deque & smoothing)
            # --------------------
            left_arm_angle = self.angle(left_sh, left_elb, left_wri)
            right_arm_angle = self.angle(right_sh, right_elb, right_wri)

            # push current angles into per-person history and compute averages
            self.prev_arm_angles[pid].append((left_arm_angle, right_arm_angle))
            avg_left, avg_right = self._avg_angle(pid)

            punch = False
            slap = False
            kick = False
            push = False
            fight = False

            # PUNCH = previous angle small (<80) and now large (>150) using averaged history
            if len(self.prev_arm_angles[pid]) >= 2:
                # previous entry (not current) for quick-change detection
                try:
                    prev_left, prev_right = self.prev_arm_angles[pid][-2]
                except Exception:
                    prev_left, prev_right = None, None

                if prev_left is not None and prev_left < 80 and avg_left > 150:
                    punch = True
                if prev_right is not None and prev_right < 80 and avg_right > 150:
                    punch = True

            # SLAP: lateral wrist offset relative to elbow with smaller angle
            try:
                if abs(left_wri[0] - left_elb[0]) > 40 and left_arm_angle < 90:
                    slap = True
                if abs(right_wri[0] - right_elb[0]) > 40 and right_arm_angle < 90:
                    slap = True
            except Exception:
                pass

            # KICK
            try:
                if left_knee[1] < left_hip[1] - 30:
                    kick = True
                if right_knee[1] < right_hip[1] - 30:
                    kick = True
            except Exception:
                pass

            # PUSH = both arms extended
            if left_arm_angle > 150 and right_arm_angle > 150:
                push = True

            # FIGHT = repeated quick moves (use per-person last_move_time)
            if punch or slap:
                now = time.time()
                if now - self.last_move_time[pid] < 1.0:
                    fight = True
                self.last_move_time[pid] = now

            # Return priorities
            if fight:
                return "Fight"
            if punch:
                return "Punch"
            if slap:
                return "Slap"
            if kick:
                return "Kick"
            if push:
                return "Push"

        return action

# -------------------------
# Main Detector class (integrates SlowFast + FER + pose)
# -------------------------
class Detector:
    def __init__(self, model, classnames):
        self.model = model
        self.classnames = classnames
        # keep heuristics as fallback but primary signal comes from pretrained SlowFast
        self.action_detector = ActionDetector()
        self.slowfast_buffer = deque(maxlen=64)  # store RGB frames
        self._sf_counter = 0
        self.slowfast_agg_count = 0           # consecutive aggression windows counter
        self.SLOWFAST_AGG_THRESHOLD = 2       # require this many consecutive windows

        # Alert cooldowns (to avoid continuous beeping)
        self.last_alert_time = {
            'fall': 0.0,
            'unsafe_pose': 0.0,
            'aggression': 0.0,
            'selfharm': 0.0
        }
        self.ALERT_COOLDOWN = 3.0  # seconds between repeated alerts of same type

        # self-harm detection: count frames where hand is near head
        self.selfharm_counter = defaultdict(lambda: 0)
        self.SELFHARM_THRESHOLD = 3    # number of frames in a short window to flag

    def _annotate(self, frame, callback=None):
        fall_detected = False
        unsafe_detected = False
        action_detected = None
        emotion_label = None
        slowfast_label = None
        slowfast_score = None
        agg_alert = False
        selfharm_detected = False
        aggression_action = None

        now = time.time()

        # -------------------- FALL DETECTION (object-detection) --------------------
        results = self.model(frame)
        for info in results:
            for box in info.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.classnames[cls]
                    conf = math.ceil(confidence * 100)

                    height = y2 - y1
                    width = x2 - x1

                    if conf > 80 and class_name == 'person':
                        cvzone.cornerRect(frame, [x1, y1, width, height])

                        # your prior rule: if height < width -> fallen
                        if height < width:
                            fall_detected = True
                            cvzone.putTextRect(frame, 'Fall Detected', [10, 40], scale=2, colorR=(0, 0, 255))
                except Exception:
                    # ignore any parse errors
                    pass

        # -------------------- UNSAFE POSE + ACTION (pose model) --------------------
        try:
            pose_results = pose_model(frame)
            for r in pose_results:
                keypoints = r.keypoints
                if keypoints is None:
                    continue

                for kp_idx, kp in enumerate(keypoints):
                    pts = kp.xy[0].tolist()

                    # convert to (x,y) ignoring confidence values if present
                    def as_xy(p):
                        return (p[0], p[1]) if len(p) >= 2 else (0.0, 0.0)

                    # safe extraction
                    rw = as_xy(pts[9]) if len(pts) > 9 else (0.0, 0.0)
                    lw = as_xy(pts[10]) if len(pts) > 10 else (0.0, 0.0)
                    rs = as_xy(pts[6]) if len(pts) > 6 else (0.0, 0.0)
                    ls = as_xy(pts[5]) if len(pts) > 5 else (0.0, 0.0)
                    nose = as_xy(pts[0]) if len(pts) > 0 else (0.0, 0.0)

                    def d(a, b):
                        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

                    # Unsafe pose: wrist near shoulder (could indicate constrained posture)
                    if d(rw, rs) < 60 or d(lw, ls) < 60:
                        unsafe_detected = True
                        cvzone.putTextRect(frame, 'Unsafe Pose Detected', [10, 90], scale=2, colorR=(255, 0, 0))

                    # Self-harm heuristic:
                    # if a hand is near nose/head repeatedly -> self-harm / self-slapping
                    hand_near_head = (d(rw, nose) < 50) or (d(lw, nose) < 50)
                    if hand_near_head:
                        # increment per-person counter (use kp_idx as temporary id)
                        self.selfharm_counter[kp_idx] += 1
                    else:
                        # decay the counter slowly
                        self.selfharm_counter[kp_idx] = max(0, self.selfharm_counter[kp_idx] - 1)

                    if self.selfharm_counter[kp_idx] >= self.SELFHARM_THRESHOLD:
                        selfharm_detected = True
                        # annotate but DO NOT beep/callback (per requirement)
                        cvzone.putTextRect(frame, 'Self-harm Detected', [10, 320], scale=1.8, colorR=(0, 0, 255))

                # Action detection (pose heuristics fallback)
                action_detected = self.action_detector.detect_action(keypoints)

        except Exception:
            pass

        # -------------------- Emotion detection (FER) --------------------
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emotion_label = emotion_detector_wrapper.detect_emotion(rgb)
            if emotion_label:
                cvzone.putTextRect(frame, f"Emotion: {emotion_label}", [10, 170], scale=1.5, colorR=(255, 200, 0))
        except Exception:
            pass

        # -------------------- SlowFast action recognition (pretrained) --------------------
        try:
            # append RGB for slowfast
            self.slowfast_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._sf_counter += 1
            if slowfast_model is not None and self._sf_counter >= SLOWFAST_STRIDE and len(self.slowfast_buffer) >= SLOWFAST_CLIP_LEN:
                clip = list(self.slowfast_buffer)[-SLOWFAST_CLIP_LEN:]
                preds = slowfast_predict(clip, topk=3)
                if preds:
                    slowfast_label, slowfast_score = preds[0]
                    # show predicted action (non-aggression labels still shown)
                    cvzone.putTextRect(frame, f"SF: {slowfast_label} ({slowfast_score:.2f})", [10, 230], scale=1.4, colorR=(0, 150, 255))

                    # Aggression mapping (kept internally but suppressed on screen)
                    if any(k in slowfast_label.lower() for k in AGGR_KEYWORDS) and slowfast_score > 0.45:
                        self.slowfast_agg_count += 1
                        aggression_action = slowfast_label
                    else:
                        self.slowfast_agg_count = 0

                    if self.slowfast_agg_count >= self.SLOWFAST_AGG_THRESHOLD:
                        agg_alert = True
                        # NOTE: aggression annotation removed per request (no putTextRect here)
                        self.slowfast_agg_count = 0

                self._sf_counter = 0
        except Exception:
            pass

        # -------------------- Decide alerts and beep (with cooldowns) --------------------
        # Fall alert (PLAY BEEP + callback)
        if fall_detected and (now - self.last_alert_time['fall'] >= self.ALERT_COOLDOWN):
            play_beep(freq=1200, length_ms=350)
            # big red overlay
            cvzone.putTextRect(frame, "ALERT: Fall Detected", [300, 60], scale=2.5, colorR=(0, 0, 255))
            self.last_alert_time['fall'] = now
            if callback:
                callback('fall', {'time': now})

        # Unsafe pose -> alert (PLAY BEEP + callback)
        if unsafe_detected and (now - self.last_alert_time['unsafe_pose'] >= self.ALERT_COOLDOWN):
            play_beep(freq=900, length_ms=300)
            cvzone.putTextRect(frame, "ALERT: Unsafe Pose Detected", [300, 120], scale=2.2, colorR=(0, 0, 255))
            self.last_alert_time['unsafe_pose'] = now
            if callback:
                callback('unsafe_pose', {'time': now})

        # Self-harm alert: annotate only, NO beep/callback
        if selfharm_detected:
            # already annotated above; optionally show subtle overlay
            cvzone.putTextRect(frame, "Note: Self-harm pattern observed", [300, 180], scale=1.2, colorR=(80, 80, 255))

        # Aggression detection suppressed entirely — no annotation, no beep, no callback
        # intentionally left blank

        # -------------------- ACTION TEXT (pose heuristics fallback) --------------------
        if action_detected:
            cvzone.putTextRect(frame, f"Action (pose): {action_detected}", [10, 140], scale=2, colorR=(0, 255, 0))
        elif slowfast_label:
            # if pose didn't detect but SlowFast did, show SlowFast label as action
            cvzone.putTextRect(frame, f"Action: {slowfast_label}", [10, 140], scale=2, colorR=(0, 255, 0))

        return frame

    # ---------------- CAMERA STREAM ----------------
    def camera_frame_generator(self, camera_index=0, callback=None):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Camera not opened")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (980, 740))
            frame = self._annotate(frame, callback=callback)
            ret2, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                jpeg.tobytes() + b'\r\n')

        cap.release()

    # ---------------- VIDEO STREAM ----------------
    def video_frame_generator(self, video_path, callback=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Video not opened:", video_path)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (980, 740))
            frame = self._annotate(frame, callback=callback)
            ret2, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                jpeg.tobytes() + b'\r\n')

        cap.release()

# ------------------ Create detector instance ------------------
detector = Detector(model, classnames)
__all__ = ['detector']

# If run directly, open a quick camera preview (optional)
if __name__ == "__main__":
    det = detector
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (980, 740))
            out = det._annotate(frame)
            cv2.imshow("Detector", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
