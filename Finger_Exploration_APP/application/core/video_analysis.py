import cv2
import torch
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict


# ─── Video reader ─────────────────────────────────────────────────────────────

class VideoProcessing:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30

    def frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame, rgb, frame.shape[:2]

    def release(self):
        self.cap.release()


# ─── Hand detector ────────────────────────────────────────────────────────────

class HandLandmarker:
    def __init__(self, num_hands, min_hand_detection_confidence,
                 min_hand_presence_confidence, min_tracking_confidence):
        BaseOptions        = mp.tasks.BaseOptions
        HandLandmarker     = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode  = mp.tasks.vision.RunningMode

        model_path = "assets/hand_landmarker.task"

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path(model_path).resolve())),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def process_frame(self, rgb_frame, timestamp):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect_for_video(mp_image, int(timestamp))

    def close(self):
        self.landmarker.close()


# ─── YOLO OBB detector ────────────────────────────────────────────────────────

class YOLO_detection:
    def __init__(self, model_path):
        self.model  = YOLO(model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def process_frame(self, frame):
        return self.model(frame, device=self.device)[0]


# ─── Constants ────────────────────────────────────────────────────────────────

INDEX_FINGER_TIP = 8   # MediaPipe landmark index for index fingertip
GRID_SIZE        = 30  # cells per axis for heatmap / direction accumulation


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def get_index_fingers(hand_result, w, h):
    """
    Return (left_pt, right_pt) pixel coords of index fingertips.
    Uses MediaPipe handedness labels. Either value is None when that hand is
    not detected or its classification is unavailable.
    """
    left_pt = right_pt = None

    if not hand_result.hand_landmarks:
        return left_pt, right_pt

    for hand_idx, hand in enumerate(hand_result.hand_landmarks):
        lm = hand[INDEX_FINGER_TIP]
        px = int(lm.x * w)
        py = int(lm.y * h)

        label = "Unknown"
        if (hand_result.handedness
                and hand_idx < len(hand_result.handedness)
                and hand_result.handedness[hand_idx]):
            label = hand_result.handedness[hand_idx][0].category_name

        if label == "Left":
            left_pt = (px, py)
        elif label == "Right":
            right_pt = (px, py)

    return left_pt, right_pt


def get_polygons(yolo_frame):
    if not hasattr(yolo_frame, "obb") or yolo_frame.obb is None:
        return []
    return [
        (obb.cpu().numpy().reshape(4, 2),
         float(yolo_frame.obb.conf[i].cpu().item()))
        for i, obb in enumerate(yolo_frame.obb.xyxyxyxy)
    ]


def select_best_polygon(polygons):
    return max(polygons, key=lambda x: x[1]) if polygons else None


def map_fingertip_to_bb(px, py, polygon):
    """Map pixel coords to normalised OBB coordinates in [0, 1]²."""
    if cv2.pointPolygonTest(polygon, (float(px), float(py)), False) < 0:
        return None
    p0, p1, p3 = polygon[0], polygon[1], polygon[3]
    x_axis = p1 - p0
    y_axis = p3 - p0
    x_len  = np.linalg.norm(x_axis)
    y_len  = np.linalg.norm(y_axis)
    if x_len == 0 or y_len == 0:
        return None
    vec = np.array([px, py], dtype=np.float64) - p0
    return (float(np.dot(vec, x_axis) / x_len ** 2),
            float(np.dot(vec, y_axis) / y_len ** 2))


# ─── Region helpers ───────────────────────────────────────────────────────────

def load_regions(region_json_path):
    with open(region_json_path, "r") as f:
        return json.load(f)


def get_region_label(x, y, regions):
    for label, (x1, y1, x2, y2) in regions.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return label
    return None


# ─── Cohort accumulator ───────────────────────────────────────────────────────

def _build_accumulator():
    return {
        "left_trajectories":   [],   # list of lists of (x, y) per video
        "right_trajectories":  [],
        "heatmap_grid":        np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64),
        "direction_sum":       np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float64),
        "direction_count":     np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64),
        "transition_counts":   defaultdict(lambda: defaultdict(int)),
        "region_first_visits": defaultdict(list),   # {region: [ts_s, ...]}
        "region_dwell_frames": defaultdict(int),    # {region: frame_count}
        "total_duration_s":    0.0,
        "video_count":         0,
    }


def _grid_idx(v):
    return min(GRID_SIZE - 1, max(0, int(v * GRID_SIZE)))


def _accumulate(acc, video_data):
    """Merge one video's processed data into the shared cohort accumulator."""
    left  = video_data["left_traj"]
    right = video_data["right_traj"]

    if left:
        acc["left_trajectories"].append(left)
    if right:
        acc["right_trajectories"].append(right)

    # Heatmap — bin every coordinate
    for xn, yn in left + right:
        acc["heatmap_grid"][_grid_idx(yn), _grid_idx(xn)] += 1

    # Direction — accumulate frame-to-frame velocity vectors per grid cell
    for traj in (left, right):
        for i in range(len(traj) - 1):
            x0, y0 = traj[i]
            x1, y1 = traj[i + 1]
            yi, xi = _grid_idx(y0), _grid_idx(x0)
            acc["direction_sum"][yi, xi, 0] += x1 - x0
            acc["direction_sum"][yi, xi, 1] += y1 - y0
            acc["direction_count"][yi, xi]  += 1

    # Region transitions
    for seq in (video_data["left_region_seq"], video_data["right_region_seq"]):
        for i in range(len(seq) - 1):
            acc["transition_counts"][seq[i][0]][seq[i + 1][0]] += 1

    # First-visit timestamps
    for region, ts in video_data["first_visits"].items():
        acc["region_first_visits"][region].append(ts)

    # Dwell frame counts
    for region, _ in video_data["left_region_seq"] + video_data["right_region_seq"]:
        acc["region_dwell_frames"][region] += 1

    acc["total_duration_s"] += video_data["duration_s"]
    acc["video_count"]      += 1


def _accumulate_one_hand(acc, video_data, hand):
    """Merge one video's data for a single hand into a per-hand accumulator."""
    traj = video_data[f"{hand}_traj"]
    seq  = video_data[f"{hand}_region_seq"]
    fv   = video_data[f"{hand}_first_visits"]

    if traj:
        acc[f"{hand}_trajectories"].append(traj)

    for xn, yn in traj:
        acc["heatmap_grid"][_grid_idx(yn), _grid_idx(xn)] += 1

    for i in range(len(traj) - 1):
        x0, y0 = traj[i]
        x1, y1 = traj[i + 1]
        yi, xi = _grid_idx(y0), _grid_idx(x0)
        acc["direction_sum"][yi, xi, 0] += x1 - x0
        acc["direction_sum"][yi, xi, 1] += y1 - y0
        acc["direction_count"][yi, xi]  += 1

    for i in range(len(seq) - 1):
        acc["transition_counts"][seq[i][0]][seq[i + 1][0]] += 1

    for region, ts in fv.items():
        acc["region_first_visits"][region].append(ts)

    for region, _ in seq:
        acc["region_dwell_frames"][region] += 1

    acc["total_duration_s"] += video_data["duration_s"]
    acc["video_count"]      += 1


# ─── Single-video processor ───────────────────────────────────────────────────

def _process_one_video(video_path, hand_model, yolo_model, regions,
                       timestamp_offset_ms=0.0):
    """
    Run detection on one video and return cohort-ready data.
    Nothing is written to disk.

    timestamp_offset_ms: cumulative offset added to every MediaPipe timestamp so
        the hand-landmarker sees a strictly monotonically increasing sequence
        across all videos processed by the same model instance.
        Region / first-visit timestamps are kept video-local (seconds from
        the start of *this* video) for meaningful cohort averaging.
    """
    vp = VideoProcessing(video_path)

    left_traj,  right_traj  = [], []
    left_region_seq, right_region_seq = [], []
    first_visits             = {}
    left_first_visits        = {}
    right_first_visits       = {}
    prev_left_rgn = prev_right_rgn = None
    local_ms  = 0.0   # video-local timestamp in ms  (for region analysis)
    duration_s = 0.0

    for frame, rgb, (h, w) in vp.frames():
        # MediaPipe needs a globally increasing integer timestamp
        mediapipe_ts = int(timestamp_offset_ms + local_ms)
        hand_res = hand_model.process_frame(rgb, mediapipe_ts)
        yolo_res = yolo_model.process_frame(frame)
        polygons = get_polygons(yolo_res)
        best     = select_best_polygon(polygons)
        ts_s     = local_ms / 1000.0   # video-local seconds for region events

        if best is not None:
            polygon, _ = best
            left_pt, right_pt = get_index_fingers(hand_res, w, h)

            # ── left index ───────────────────────────────────────────────────
            if left_pt is not None:
                mapped = map_fingertip_to_bb(left_pt[0], left_pt[1], polygon)
                if mapped:
                    xn, yn = mapped
                    left_traj.append((xn, yn))
                    region = get_region_label(xn, yn, regions)
                    if region:
                        first_visits.setdefault(region, ts_s)
                        left_first_visits.setdefault(region, ts_s)
                        if region != prev_left_rgn:
                            left_region_seq.append((region, ts_s))
                            prev_left_rgn = region

            # ── right index ──────────────────────────────────────────────────
            if right_pt is not None:
                mapped = map_fingertip_to_bb(right_pt[0], right_pt[1], polygon)
                if mapped:
                    xn, yn = mapped
                    right_traj.append((xn, yn))
                    region = get_region_label(xn, yn, regions)
                    if region:
                        first_visits.setdefault(region, ts_s)
                        right_first_visits.setdefault(region, ts_s)
                        if region != prev_right_rgn:
                            right_region_seq.append((region, ts_s))
                            prev_right_rgn = region

        duration_s  = ts_s
        local_ms   += 1000.0 / vp.fps

    vp.release()

    return {
        "left_traj":          left_traj,
        "right_traj":         right_traj,
        "left_region_seq":    left_region_seq,
        "right_region_seq":   right_region_seq,
        "first_visits":       first_visits,
        "left_first_visits":  left_first_visits,
        "right_first_visits": right_first_visits,
        "duration_s":         duration_s,
        "duration_ms":        local_ms,   # total ms consumed — used to advance offset
    }


# ─── Summary builder ──────────────────────────────────────────────────────────

def _build_summary(acc, video_folder):
    avg_fv = {
        r: sum(ts) / len(ts)
        for r, ts in acc["region_first_visits"].items()
    }
    sorted_rg = sorted(avg_fv.items(), key=lambda kv: kv[1])

    total_transitions = sum(
        sum(dsts.values()) for dsts in acc["transition_counts"].values()
    )

    return {
        "video_folder":                     str(video_folder),
        "total_videos_processed":           acc["video_count"],
        "total_exploration_time_s":         round(acc["total_duration_s"], 2),
        "avg_exploration_time_per_video_s": round(
            acc["total_duration_s"] / max(acc["video_count"], 1), 2),
        "total_transitions":                total_transitions,
        "consensus_region_sequence":        " -> ".join(r for r, _ in sorted_rg),
        "region_avg_first_visit_s":         {r: round(t, 3) for r, t in sorted_rg},
        "region_dwell_frames":              dict(acc["region_dwell_frames"]),
    }


def _write_cohort_outputs(acc, out_dir, regions, video_folder,
                          tactile_image_path=None,
                          left_transition_counts=None,
                          right_transition_counts=None):
    """Generate all 5 plots and cohort_summary.json into out_dir."""
    from core.plots import (
        generate_cumulative_trajectory,
        generate_cumulative_heatmap,
        generate_cumulative_direction,
        generate_cumulative_transition_graph,
        generate_cumulative_spatial_transition_graph,
    )

    generate_cumulative_trajectory(
        acc["left_trajectories"],
        acc["right_trajectories"],
        out_dir / "cumulative_trajectory.png",
    )
    print("  Saved  cumulative_trajectory.png")

    generate_cumulative_heatmap(
        acc["heatmap_grid"],
        acc["video_count"],
        out_dir / "cumulative_heatmap.png",
    )
    print("  Saved  cumulative_heatmap.png")

    _avg_fv    = {r: sum(ts) / len(ts)
                  for r, ts in acc["region_first_visits"].items()}
    _sorted_rg = sorted(_avg_fv.items(), key=lambda kv: kv[1])
    _start_rgn = _sorted_rg[0][0]  if _sorted_rg else None
    _end_rgn   = _sorted_rg[-1][0] if _sorted_rg else None

    generate_cumulative_direction(
        acc["direction_sum"],
        acc["direction_count"],
        out_dir / "cumulative_direction.png",
        regions=regions,
        start_region=_start_rgn,
        end_region=_end_rgn,
    )
    print("  Saved  cumulative_direction.png")

    generate_cumulative_transition_graph(
        acc["transition_counts"],
        out_dir / "cumulative_transition_graph.png",
        regions=regions,
    )
    print("  Saved  cumulative_transition_graph.png")

    generate_cumulative_spatial_transition_graph(
        acc["transition_counts"],
        acc["region_dwell_frames"],
        regions,
        out_dir / "cumulative_spatial_transition_graph.png",
        background_image_path=tactile_image_path,
        left_transition_counts=left_transition_counts,
        right_transition_counts=right_transition_counts,
    )
    print("  Saved  cumulative_spatial_transition_graph.png")

    summary = _build_summary(acc, video_folder)
    with open(out_dir / "cohort_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("  Saved  cohort_summary.json")


# ─── Public entry point ───────────────────────────────────────────────────────

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def run_batch_video_analysis(
    video_folder,
    model_path,
    region_json_path,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    tactile_image_path=None,
):
    """
    Cohort-level batch analysis pipeline.

    Processes every video in *video_folder* sequentially, accumulates all
    index-finger tracking data in memory, then writes outputs to 3 subfolders:
        both_hands/   — combined left + right index finger data
        left_hand/    — left index finger data only
        right_hand/   — right index finger data only

    Each subfolder contains:
        cohort_summary.json
        cumulative_trajectory.png
        cumulative_heatmap.png
        cumulative_direction.png
        cumulative_transition_graph.png
        cumulative_spatial_transition_graph.png

    Pass tactile_image_path to overlay the graphic photo on the spatial graph.
    """
    hand_model = HandLandmarker(
        num_hands,
        min_hand_detection_confidence,
        min_hand_presence_confidence,
        min_tracking_confidence,
    )
    yolo_model = YOLO_detection(model_path)
    regions    = load_regions(region_json_path)

    video_files = sorted(
        f for f in Path(video_folder).iterdir()
        if f.suffix.lower() in _VIDEO_EXTS
    )

    if not video_files:
        print("No video files found in the selected folder.")
        hand_model.close()
        return

    print(f"Cohort: {len(video_files)} video(s) detected — starting pipeline…")

    acc       = _build_accumulator()
    acc_left  = _build_accumulator()
    acc_right = _build_accumulator()

    # MediaPipe VIDEO mode requires a single, globally increasing timestamp stream.
    # We advance this offset by each video's duration + a 100 ms inter-video gap
    # so subsequent videos never send a timestamp earlier than the previous one.
    next_ts_offset_ms = 0.0

    for i, vf in enumerate(video_files, 1):
        print(f"  [{i}/{len(video_files)}]  {vf.name}")
        try:
            data = _process_one_video(
                str(vf), hand_model, yolo_model, regions,
                timestamp_offset_ms=next_ts_offset_ms,
            )
            _accumulate(acc, data)
            _accumulate_one_hand(acc_left,  data, "left")
            _accumulate_one_hand(acc_right, data, "right")
            # Advance the offset: this video's total duration + 100 ms guard gap
            next_ts_offset_ms += data["duration_ms"] + 100.0
            print(f"    left={len(data['left_traj'])}  "
                  f"right={len(data['right_traj'])}  "
                  f"dur={data['duration_s']:.1f}s")
        except Exception as exc:
            print(f"    FAILED: {exc}")
            import traceback; traceback.print_exc()

    hand_model.close()

    # ── Output directory with 3 subfolders ───────────────────────────────────
    now       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir   = Path(video_folder).parent / f"cohort_{Path(video_folder).name}_{now}"
    dir_both  = out_dir / "both_hands"
    dir_left  = out_dir / "left_hand"
    dir_right = out_dir / "right_hand"
    for d in (dir_both, dir_left, dir_right):
        d.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput → {out_dir}")

    # ── Generate plots for each accumulator ──────────────────────────────────
    for label, sub_acc, sub_dir, l_tc, r_tc in (
        ("Both Hands",  acc,       dir_both,  acc_left["transition_counts"], acc_right["transition_counts"]),
        ("Left Hand",   acc_left,  dir_left,  None,                          None),
        ("Right Hand",  acc_right, dir_right, None,                          None),
    ):
        print(f"\n── {label} ──")
        try:
            _write_cohort_outputs(
                sub_acc, sub_dir, regions, video_folder,
                tactile_image_path=tactile_image_path,
                left_transition_counts=l_tc,
                right_transition_counts=r_tc,
            )
        except Exception as exc:
            import traceback
            print(f"  Warning: output generation failed for {label} — {exc}")
            traceback.print_exc()

    print(f"\n✅  Cohort complete — {acc['video_count']} video(s) processed.")
    print(f"    Output: {out_dir}")
