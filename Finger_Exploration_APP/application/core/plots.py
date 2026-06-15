import colorsys
import json
import os
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def generate_trajectory_plot(clean_json_path, output_path):
    with open(clean_json_path, "r") as f:
        data = json.load(f)

    if not data:
        return

    xs = [point["x"] for point in data]
    ys = [point["y"] for point in data]

    plt.figure(figsize=(6, 6))

    plt.plot(xs, ys, marker="o")

    plt.scatter(xs[0], ys[0])
    plt.text(xs[0], ys[0], "START")

    plt.scatter(xs[-1], ys[-1])
    plt.text(xs[-1], ys[-1], "END")

    plt.gca().invert_yaxis()
    plt.title("Finger Movement Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.savefig(output_path)
    plt.close()


def generate_direction_plot(clean_json_path, output_path):
    with open(clean_json_path, "r") as f:
        data = json.load(f)

    if len(data) < 2:
        return

    xs = [point["x"] for point in data]
    ys = [point["y"] for point in data]

    plt.figure(figsize=(6, 6))

    plt.quiver(
        xs[:-1],
        ys[:-1],
        [xs[i + 1] - xs[i] for i in range(len(xs) - 1)],
        [ys[i + 1] - ys[i] for i in range(len(ys) - 1)],
        angles="xy",
        scale_units="xy",
        scale=1
    )

    plt.gca().invert_yaxis()
    plt.title("Finger Movement Direction")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.savefig(output_path)
    plt.close()


def generate_heatmap_plot(clean_json_path, output_path):
    with open(clean_json_path, "r") as f:
        data = json.load(f)

    if len(data) < 2:
        return

    xs = [point["x"] for point in data]
    ys = [point["y"] for point in data]

    plt.figure(figsize=(6, 6))

    plt.hist2d(xs, ys, bins=30)
    plt.gca().invert_yaxis()
    plt.title("Finger Exploration Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()

    plt.savefig(output_path)
    plt.close()


def generate_transition_graph(analysis_json_path, output_path):
    with open(analysis_json_path, "r") as f:
        data = json.load(f)

    transitions = data["transitions"]

    if not transitions:
        return

    G = nx.DiGraph()

    for transition in transitions:
        src, dst = transition.split("->")

        if G.has_edge(src, dst):
            G[src][dst]["weight"] += 1
        else:
            G.add_edge(src, dst, weight=1)

    plt.figure(figsize=(6, 6))

    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, "weight")

    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Transition Graph")

    plt.savefig(output_path)
    plt.close()


def generate_spatial_transition_graph(
    analysis_json_path,
    region_json_path,
    output_path,
    background_image_path=None
):
    with open(analysis_json_path, "r") as f:
        data = json.load(f)

    with open(region_json_path, "r") as f:
        regions = json.load(f)

    sequence = data["sequence"]

    if len(sequence) < 2:
        return

    centers = {}

    for name, (x1, y1, x2, y2) in regions.items():
        centers[name] = (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

    transition_count = defaultdict(int)

    for i in range(len(sequence) - 1):
        key = (sequence[i], sequence[i + 1])
        transition_count[key] += 1

    plt.figure(figsize=(8, 8))

    if background_image_path:
        img = cv2.imread(background_image_path)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img, extent=[0, 1, 1, 0])

    for name, (x1, y1, x2, y2) in regions.items():
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2
        )

        plt.gca().add_patch(rect)

        cx, cy = centers[name]
        plt.text(cx, cy, name, ha="center", va="center")

    for (src, dst), count in transition_count.items():
        x1, y1 = centers[src]
        x2, y2 = centers[dst]

        dx = x2 - x1
        dy = y2 - y1

        plt.arrow(
            x1,
            y1,
            dx,
            dy,
            length_includes_head=True,
            head_width=0.015,
            linewidth=1 + count,
            alpha=0.8
        )

    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.title("Spatial Transition Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()

    plt.savefig(output_path)
    plt.close()


# ─── New functions for combined output format ─────────────────────────────────

def generate_scan_path_plot(combined_output, out_path, template_img_path=None):
    """
    Build a color-gradient scan-path trajectory from the combined video output list.

    combined_output: list of frame dicts produced by process_video()
    out_path:        destination file path for trajectory.png
    template_img_path: optional path to master template image used as background
    """
    points = []  # (x_norm, y_norm, timestamp_ms, frame_idx)
    for frame_data in combined_output:
        frame_idx = frame_data.get("frame", 0)
        ts_ms     = frame_data.get("timestamp_ms", 0)
        for inter in frame_data.get("interactions", []):
            norm = inter.get("norm", [])
            if len(norm) >= 2:
                points.append((float(norm[0]), float(norm[1]), ts_ms, frame_idx))

    if len(points) < 2:
        return

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    n  = len(points)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a12")
    ax.set_facecolor("#0f0f1a")

    # Optional background template image
    if template_img_path and os.path.exists(str(template_img_path)):
        bg = cv2.imread(str(template_img_path))
        if bg is not None:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            ax.imshow(bg, extent=[0, 1, 0, 1], aspect="auto", alpha=0.35)

    # Draw colored path segments — jet colormap shifts blue→red over time
    cmap = plt.cm.jet
    for i in range(n - 1):
        color = cmap(i / max(n - 1, 1))
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color=color, linewidth=1.6, alpha=0.85, solid_capstyle="round")

    # Sequence-index markers every ~15 steps along the path
    step = max(1, n // 15)
    for i in range(0, n, step):
        ax.text(xs[i], ys[i], str(i),
                fontsize=6, color="#ffffff99", ha="center", va="center", zorder=4)

    # Start (green ●) and End (red ✕)
    ax.scatter(xs[0],  ys[0],  c="#22c55e", s=160, zorder=6, marker="o",
               edgecolors="white", linewidths=0.8, label="Start")
    ax.scatter(xs[-1], ys[-1], c="#ef4444", s=160, zorder=6, marker="X",
               linewidths=1.5, label="End")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("X (normalized)", color="#9090b8", fontsize=10)
    ax.set_ylabel("Y (normalized)", color="#9090b8", fontsize=10)
    ax.set_title("Scan Path Trajectory",
                 color="#e8e8f0", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="#606080")
    for spine in ax.spines.values():
        spine.set_edgecolor("#252538")

    ax.legend(loc="upper right", facecolor="#1a1a28",
              edgecolor="#252538", labelcolor="#e8e8f0", fontsize=9)

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Point index  (earlier → later)", pad=0.02)
    cbar.ax.yaxis.label.set_color("#9090b8")
    cbar.ax.tick_params(colors="#606080")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    plt.close(fig)


def generate_heatmap_from_combined(combined_output, out_path):
    """
    Build a 2-D interaction-frequency heatmap from the combined video output list.

    combined_output: list of frame dicts produced by process_video()
    out_path:        destination file path for heatmap.png
    """
    xs, ys = [], []
    for frame_data in combined_output:
        for inter in frame_data.get("interactions", []):
            norm = inter.get("norm", [])
            if len(norm) >= 2:
                xs.append(float(norm[0]))
                ys.append(float(norm[1]))

    if len(xs) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a12")
    ax.set_facecolor("#0f0f1a")

    h = ax.hist2d(xs, ys, bins=30, cmap="hot", density=False)
    cbar = fig.colorbar(h[3], ax=ax, label="Interaction frequency", pad=0.02)
    cbar.ax.yaxis.label.set_color("#9090b8")
    cbar.ax.tick_params(colors="#606080")

    ax.set_title("Finger Exploration Heatmap",
                 color="#e8e8f0", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (normalized)", color="#9090b8", fontsize=10)
    ax.set_ylabel("Y (normalized)", color="#9090b8", fontsize=10)
    ax.tick_params(colors="#606080")
    for spine in ax.spines.values():
        spine.set_edgecolor("#252538")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    plt.close(fig)


def generate_animated_trajectory(combined_output, out_path, template_img_path=None,
                                  trail_seconds=2.0, anim_fps=8):
    """
    Generate an animated GIF showing a finger-pointer moving over the graphic.

    The trail fades red (newest) → blue (oldest) over trail_seconds, then
    disappears gradually — never abruptly.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    points = []
    for frame_data in combined_output:
        ts_ms = float(frame_data.get("timestamp_ms", 0))
        for inter in frame_data.get("interactions", []):
            norm = inter.get("norm", [])
            if len(norm) >= 2:
                points.append((float(norm[0]), float(norm[1]), ts_ms))

    if len(points) < 5:
        return

    trail_ms          = trail_seconds * 1000.0
    start_ts          = points[0][2]
    end_ts            = points[-1][2]
    total_ms          = max(1.0, end_ts - start_ts)

    # Cap at 180 frames to keep file size manageable
    n_frames          = min(180, max(20, int(total_ms / (1000.0 / anim_fps))))
    frame_interval_ms = total_ms / n_frames

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0a0a12")
    ax.set_facecolor("#0f0f1a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("X", color="#9090b8", fontsize=8)
    ax.set_ylabel("Y", color="#9090b8", fontsize=8)
    ax.tick_params(colors="#606080", labelsize=7)
    ax.set_title("Finger Scan Path — Animated", color="#e8e8f0",
                 fontsize=10, fontweight="bold", pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#252538")

    if template_img_path and os.path.exists(str(template_img_path)):
        bg = cv2.imread(str(template_img_path))
        if bg is not None:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            ax.imshow(bg, extent=[0, 1, 0, 1], aspect="auto", alpha=0.35, zorder=0)

    time_text   = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                          color="#e8e8f0", fontsize=9, va="top", zorder=10)
    pointer_dot = ax.scatter([], [], c="#ffffff", s=90, zorder=8,
                              edgecolors="#e8e8f0", linewidths=0.8)
    trail_artists = []

    def update(frame_num):
        nonlocal trail_artists
        for a in trail_artists:
            try:
                a.remove()
            except ValueError:
                pass
        trail_artists = []

        current_ts   = start_ts + frame_num * frame_interval_ms
        window_start = current_ts - trail_ms

        window_pts = [(x, y, ts) for x, y, ts in points
                      if window_start <= ts <= current_ts]

        if len(window_pts) >= 2:
            segs, colors = [], []
            for i in range(len(window_pts) - 1):
                x0, y0, ts0 = window_pts[i]
                x1, y1, _   = window_pts[i + 1]
                age_ratio   = min(1.0, (current_ts - ts0) / trail_ms)
                alpha       = max(0.04, 1.0 - age_ratio ** 0.55)
                r = max(0.0, 1.0 - age_ratio)
                b = min(1.0, age_ratio * 1.5)
                segs.append([(x0, y0), (x1, y1)])
                colors.append((r, 0.06, b, alpha))

            lc = LineCollection(segs, colors=colors,
                                linewidths=2.6, capstyle="round", zorder=2)
            ax.add_collection(lc)
            trail_artists.append(lc)
            pointer_dot.set_offsets([[window_pts[-1][0], window_pts[-1][1]]])
        else:
            pointer_dot.set_offsets(np.empty((0, 2)))

        time_text.set_text(f"t = {(current_ts - start_ts) / 1000:.1f} s")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=int(1000 / anim_fps), blit=False, repeat=True)
    try:
        anim.save(str(out_path), writer="pillow", fps=anim_fps, dpi=90,
                  savefig_kwargs={"facecolor": "#0a0a12"})
    finally:
        plt.close(fig)


# ─── Helpers for the tracking video ──────────────────────────────────────────

#: MediaPipe landmark index → human-readable finger name
FINGER_NAMES = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}


def _finger_colors():
    """Return a BGR color map keyed by (hand_id, finger_id)."""
    return {
        (0, 4):  (60,   60,  245),   # H0 Thumb  – Red
        (0, 8):  (60,  215,   60),   # H0 Index  – Green
        (0, 12): (245,  60,   60),   # H0 Middle – Blue
        (0, 16): (30,  225,  225),   # H0 Ring   – Yellow
        (0, 20): (215, 215,   30),   # H0 Pinky  – Cyan
        (1, 4):  (20,  130,  255),   # H1 Thumb  – Orange
        (1, 8):  (210,  50,  210),   # H1 Index  – Magenta
        (1, 12): (60,  240,  120),   # H1 Middle – Lime
        (1, 16): (190,  60,  170),   # H1 Ring   – Pink
        (1, 20): (170, 200,   45),   # H1 Pinky  – Teal
    }


def _get_finger_color(hand, finger, color_map):
    """Return BGR color for (hand, finger); hash-based fallback for unknowns."""
    key = (int(hand), int(finger))
    if key in color_map:
        return color_map[key]
    h_val = abs(hash(key)) % 360
    r, g, b = colorsys.hsv_to_rgb(h_val / 360.0, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _dark_canvas(size):
    """Dark grid canvas used as fallback video background."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 26)
    step = size // 10
    for i in range(0, size + 1, step):
        cv2.line(canvas, (i, 0),    (i, size),  (30, 30, 50), 1)
        cv2.line(canvas, (0, i),    (size, i),  (30, 30, 50), 1)
    return canvas


def _draw_video_legend(canvas, present_fingers, color_map, canvas_size):
    """Semi-transparent legend in the top-right corner."""
    if not present_fingers:
        return
    items  = sorted(present_fingers)
    lh     = 17
    pad    = 6
    box_w  = 106
    box_h  = len(items) * lh + pad * 2
    x0     = canvas_size - box_w - 4
    y0     = 4
    ovl    = canvas.copy()
    cv2.rectangle(ovl, (x0 - 2, y0), (canvas_size - 4, y0 + box_h), (15, 15, 30), -1)
    cv2.addWeighted(ovl, 0.65, canvas, 0.35, 0, canvas)
    for i, (h, f) in enumerate(items):
        color = _get_finger_color(h, f, color_map)
        fname = FINGER_NAMES.get(f, f"F{f}")
        cy    = y0 + pad + i * lh + lh // 2
        cv2.circle(canvas, (x0 + 6, cy), 4, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"H{h} {fname}", (x0 + 16, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)


# ─── New plot generators (work with combined output or session folder) ────────

def generate_direction_from_combined(combined_output, out_path):
    """Quiver / direction plot aggregated from the combined video output."""
    xs, ys = [], []
    for frame_data in combined_output:
        for inter in frame_data.get("interactions", []):
            norm = inter.get("norm", [])
            if len(norm) >= 2:
                xs.append(float(norm[0]))
                ys.append(float(norm[1]))

    if len(xs) < 4:
        return

    step = max(1, len(xs) // 120)   # sample for readability
    sx, sy = xs[::step], ys[::step]
    u = [sx[i + 1] - sx[i] for i in range(len(sx) - 1)]
    v = [sy[i + 1] - sy[i] for i in range(len(sy) - 1)]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a12")
    ax.set_facecolor("#0f0f1a")
    ax.quiver(sx[:-1], sy[:-1], u, v,
              angles="xy", scale_units="xy", scale=1,
              color="#0ea5e9", alpha=0.72, width=0.004)

    # Start (green ●) and End (red ✕) markers on the full-resolution series
    ax.scatter(xs[0],  ys[0],  c="#22c55e", s=180, zorder=6, marker="o",
               edgecolors="white", linewidths=0.8, label="Start")
    ax.scatter(xs[-1], ys[-1], c="#ef4444", s=180, zorder=6, marker="X",
               linewidths=1.5, label="End")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Finger Movement Direction",
                 color="#e8e8f0", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (normalized)", color="#9090b8", fontsize=10)
    ax.set_ylabel("Y (normalized)", color="#9090b8", fontsize=10)
    ax.tick_params(colors="#606080")
    for spine in ax.spines.values():
        spine.set_edgecolor("#252538")
    ax.legend(loc="upper right", facecolor="#1a1a28",
              edgecolor="#252538", labelcolor="#e8e8f0", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    plt.close(fig)


def generate_transition_graph_from_folder(session_folder, out_path):
    """Directed region-transition network graph built from all transitions.json files."""
    counts = {}
    for tp in Path(session_folder).rglob("transitions.json"):
        try:
            seq = [s["region"] for s in json.load(open(tp)).get("sequence", [])]
            for i in range(len(seq) - 1):
                k = (seq[i], seq[i + 1])
                counts[k] = counts.get(k, 0) + 1
        except Exception:
            continue

    if not counts:
        return

    G = nx.DiGraph()
    for (src, dst), w in counts.items():
        G.add_edge(src, dst, weight=w)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w   = max(weights) if weights else 1

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a12")
    ax.set_facecolor("#0f0f1a")
    pos = nx.spring_layout(G, seed=42, k=2.0)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#0ea5e9",
                           node_size=900, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="#ffffff",
                            font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#a855f7", alpha=0.75,
                           arrows=True, arrowsize=22,
                           width=[1.0 + 3.0 * w / max_w for w in weights],
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={e: G.edges[e]["weight"] for e in G.edges()},
        ax=ax, font_color="#f59e0b", font_size=8)

    ax.set_title("Region Transition Graph",
                 color="#e8e8f0", fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    plt.close(fig)


def generate_spatial_transition_from_folder(session_folder, region_json_path,
                                             out_path, template_img_path=None):
    """Spatial arrows between region centres weighted by visit frequency."""
    try:
        regions = json.load(open(region_json_path))
    except Exception:
        return

    counts = {}
    for tp in Path(session_folder).rglob("transitions.json"):
        try:
            seq = [s["region"] for s in json.load(open(tp)).get("sequence", [])]
            for i in range(len(seq) - 1):
                k = (seq[i], seq[i + 1])
                counts[k] = counts.get(k, 0) + 1
        except Exception:
            continue

    if not counts:
        return

    centers = {
        name: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for name, b in regions.items()
        if isinstance(b, (list, tuple)) and len(b) == 4
    }

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a12")
    ax.set_facecolor("#0f0f1a")

    if template_img_path and os.path.exists(str(template_img_path)):
        img = cv2.imread(str(template_img_path))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                      extent=[0, 1, 1, 0], aspect="auto", alpha=0.35)

    for name, b in regions.items():
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = b
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor="#0ea5e9", linewidth=1.5))
            if name in centers:
                cx, cy = centers[name]
                ax.text(cx, cy, name, ha="center", va="center",
                        color="#e8e8f0", fontsize=9, fontweight="bold")

    max_c = max(counts.values()) if counts else 1
    for (src, dst), c in counts.items():
        if src not in centers or dst not in centers:
            continue
        ax.annotate("", xy=centers[dst], xytext=centers[src],
                    arrowprops=dict(arrowstyle="-|>", color="#a855f7",
                                   lw=1.0 + 3.0 * c / max_c,
                                   alpha=0.35 + 0.6 * c / max_c,
                                   connectionstyle="arc3,rad=0.08"))

    ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    ax.set_title("Spatial Transition Graph",
                 color="#e8e8f0", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (normalized)", color="#9090b8", fontsize=10)
    ax.set_ylabel("Y (normalized)", color="#9090b8", fontsize=10)
    ax.tick_params(colors="#606080")
    for spine in ax.spines.values():
        spine.set_edgecolor("#252538")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    plt.close(fig)


def _extract_bg_from_video(combined_output, video_path, canvas_size):
    """
    Warp the tactile graphic out of the original video using the stored OBB polygon.

    The destination corners are flipped vertically so the warped background
    aligns with the Y-flipped coordinate system used in the tracking video
    (norm y=0 → canvas bottom, norm y=1 → canvas top).
    """
    # Find the frame with the highest-confidence polygon
    best_frame_idx = None
    best_polygon   = None
    best_conf      = -1.0

    for frame_data in combined_output:
        for inter in frame_data.get("interactions", []):
            poly = inter.get("polygon")
            conf = float(inter.get("confidence", 0.0))
            if poly and conf > best_conf:
                best_conf      = conf
                best_polygon   = np.array(poly, dtype=np.float32)
                best_frame_idx = frame_data["frame"]

    if best_frame_idx is None or best_polygon is None:
        return _dark_canvas(canvas_size)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return _dark_canvas(canvas_size)

    cs = canvas_size - 1
    # src corners follow YOLO OBB vertex order: p0 (origin), p1 (x-end), p2 (xy-end), p3 (y-end)
    # dst corners are flipped vertically so that p0 → bottom-left, p3 → top-left
    src_pts = best_polygon.astype(np.float32)
    dst_pts = np.array([
        [0,  cs],    # p0 (norm 0,0) → bottom-left
        [cs, cs],    # p1 (norm 1,0) → bottom-right
        [cs,  0],    # p2 (norm 1,1) → top-right
        [0,   0],    # p3 (norm 0,1) → top-left
    ], dtype=np.float32)

    M       = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped  = cv2.warpPerspective(frame, M, (canvas_size, canvas_size))
    return (warped.astype(np.float32) * 0.38).astype(np.uint8)


def generate_tracking_video(combined_output, out_path, template_img_path=None,
                             video_path=None,
                             trail_seconds=2.0, speed_multiplier=1.0, canvas_size=600):
    """
    Multi-finger tracking MP4.

    Each (hand, finger) pair gets a unique BGR color. A fading trail shows
    the last `trail_seconds` of movement per finger — newest = bright, oldest = dark.
    Y coordinates are flipped so the physical orientation of the tactile graphic
    (thumb at bottom) renders correctly.

    Pass `video_path` to extract and warp the tactile graphic as the background.
    The video runs at `speed_multiplier`× FPS (default 1.25×).
    """
    if not combined_output:
        return

    # ── FPS ───────────────────────────────────────────────────────────────────
    input_fps = 30.0
    if len(combined_output) > 1:
        elapsed = combined_output[-1].get("timestamp_ms", 0) / 1000.0
        if elapsed > 0:
            input_fps = max(1.0, len(combined_output) / elapsed)
    output_fps = min(60.0, max(1.0, input_fps * speed_multiplier))
    trail_ms   = trail_seconds * 1000.0

    COLORS = _finger_colors()

    # ── Background ────────────────────────────────────────────────────────────
    if video_path and os.path.exists(str(video_path)):
        bg = _extract_bg_from_video(combined_output, str(video_path), canvas_size)
    elif template_img_path and os.path.exists(str(template_img_path)):
        img = cv2.imread(str(template_img_path))
        if img is not None:
            img = cv2.resize(img, (canvas_size, canvas_size))
            bg  = (img.astype(np.float32) * 0.38).astype(np.uint8)
        else:
            bg = _dark_canvas(canvas_size)
    else:
        bg = _dark_canvas(canvas_size)

    # ── Discover which finger IDs actually appear (for legend) ────────────────
    present = set()
    for fd in combined_output:
        for inter in fd.get("interactions", []):
            present.add((int(inter.get("hand", 0)), int(inter.get("finger", 0))))

    # ── VideoWriter ───────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps,
                              (canvas_size, canvas_size))
    if not writer.isOpened():
        print(f"Warning: VideoWriter could not open {out_path}")
        return

    def _cy(y_norm):
        """Flip Y so norm=0 → canvas bottom, norm=1 → canvas top."""
        return int((1.0 - float(y_norm)) * canvas_size)

    def _cx(x_norm):
        return int(float(x_norm) * canvas_size)

    trail: deque = deque()   # (x_norm, y_norm, ts_ms, hand, finger)

    try:
        for frame_data in combined_output:
            current_ts   = float(frame_data.get("timestamp_ms", 0))
            window_start = current_ts - trail_ms

            for inter in frame_data.get("interactions", []):
                norm = inter.get("norm", [])
                if len(norm) >= 2:
                    trail.append((float(norm[0]), float(norm[1]),
                                  current_ts,
                                  int(inter.get("hand", 0)),
                                  int(inter.get("finger", 0))))

            while trail and trail[0][2] < window_start:
                trail.popleft()

            canvas = bg.copy()

            # ── Trail lines per finger ────────────────────────────────────────
            by_hf: dict = {}
            for x, y, ts, h, f in trail:
                by_hf.setdefault((h, f), []).append((x, y, ts))

            for (h, f), pts in by_hf.items():
                color = _get_finger_color(h, f, COLORS)
                for i in range(len(pts) - 1):
                    x0, y0, ts0 = pts[i]
                    x1, y1, _   = pts[i + 1]
                    age   = min(1.0, (current_ts - ts0) / trail_ms)
                    alpha = max(0.0, 1.0 - age ** 0.55)
                    fade  = tuple(max(0, int(c * alpha)) for c in color)
                    thick = max(1, int(3 * alpha))
                    cv2.line(canvas,
                             (_cx(x0), _cy(y0)),
                             (_cx(x1), _cy(y1)),
                             fade, thick, cv2.LINE_AA)

            # ── Current-frame finger dots ─────────────────────────────────────
            for inter in frame_data.get("interactions", []):
                norm = inter.get("norm", [])
                if len(norm) < 2:
                    continue
                h     = int(inter.get("hand", 0))
                f     = int(inter.get("finger", 0))
                color = _get_finger_color(h, f, COLORS)
                px    = _cx(norm[0])
                py    = _cy(norm[1])

                # Soft glow ring
                ovl = canvas.copy()
                cv2.circle(ovl, (px, py), 18, color, -1, cv2.LINE_AA)
                cv2.addWeighted(ovl, 0.20, canvas, 0.80, 0, canvas)

                # Solid dot + white outline
                cv2.circle(canvas, (px, py), 7, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)

                # Short label next to dot
                fname = FINGER_NAMES.get(f, f"F{f}")[:2]
                lx = min(px + 10, canvas_size - 55)
                ly = max(py - 9, 10)
                cv2.putText(canvas, f"H{h}-{fname}", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

            # ── HUD: timestamp ────────────────────────────────────────────────
            cv2.putText(canvas, f"t={current_ts / 1000:.2f}s  [1x]",
                        (6, canvas_size - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 180), 1, cv2.LINE_AA)

            # ── Legend ────────────────────────────────────────────────────────
            _draw_video_legend(canvas, present, COLORS, canvas_size)

            writer.write(canvas)
    finally:
        writer.release()


# ─── Shared helpers (cumulative plots) ───────────────────────────────────────

def _smooth_traj(traj, window=11, poly=2):
    """
    Savitzky-Golay smoothing on a list-of-(x,y) trajectory.
    Eliminates micro-tremors while preserving macroscopic scan-path shape.
    Falls back to the raw trajectory when it is too short to filter.
    """
    from scipy.signal import savgol_filter
    n = len(traj)
    if n < poly + 2:
        return traj
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return traj
    xs = np.array([p[0] for p in traj], dtype=float)
    ys = np.array([p[1] for p in traj], dtype=float)
    return list(zip(savgol_filter(xs, w, poly), savgol_filter(ys, w, poly)))


def _arc_label_xy(p1, p2, rad, extra=0.028):
    """
    Return the (x, y) position for a probability label that floats beside
    an arc3 arrow without intersecting the line.

    The label is placed at the arc's geometric midpoint
    (chord_midpoint + perp * rad*chord/2) plus an additional `extra` offset
    perpendicular to the chord so it clears the arrowhead.
    """
    x1, y1 = p1
    x2, y2 = p2
    mx, my  = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy  = x2 - x1, y2 - y1
    chord   = max(np.sqrt(dx ** 2 + dy ** 2), 1e-10)
    # Unit perpendicular (90° CCW of edge direction → "left of u→v")
    perp_x, perp_y = -dy / chord, dx / chord
    total_offset    = rad * chord / 2.0 + extra
    return mx + perp_x * total_offset, my + perp_y * total_offset


# ─── Cumulative cohort plot functions ─────────────────────────────────────────

_BG_DARK  = "#0a0a12"   # figure facecolor
_BG_AXES  = "#0f0f1a"   # axes facecolor  (also used as label-mask colour)
_SPINE_C  = "#252538"
_TICK_C   = "#606080"
_LABEL_C  = "#9090b8"
_TEXT_C   = "#e8e8f0"

def _save_empty_plot(out_path, title):
    """Save a dark-themed placeholder when a hand has no data to display."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=_BG_DARK)
    ax.set_facecolor(_BG_AXES)
    ax.text(0.5, 0.52, "No data available",
            ha="center", va="center",
            color=_LABEL_C, fontsize=18, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.46, "No finger tracking data was detected for this hand.",
            ha="center", va="center",
            color=_TICK_C, fontsize=11,
            transform=ax.transAxes)
    ax.set_title(title, color=_TEXT_C, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE_C)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)


# High-contrast hand colours (neon against dark backgrounds)
_LEFT_C   = "#00d4ff"   # neon cyan-blue  → left index finger
_RIGHT_C  = "#ff4060"   # coral-crimson   → right index finger


def generate_cumulative_trajectory(left_trajectories, right_trajectories, out_path):
    """
    Superimposed, smoothed index-finger scan paths for every cohort participant.

    • Each path is rendered at alpha=0.12 with a hairline width (lw=0.75) so
      common scanning highways compound to full brightness while one-off
      deviations stay near-invisible.
    • Savitzky-Golay smoothing is applied per trajectory to remove frame-jitter
      tremors before plotting.
    • Left hand → neon cyan (#00d4ff) | Right hand → coral-crimson (#ff4060)
    • Per-trajectory START (●) and END (✕) markers are overlaid at alpha=0.55
      so individual start/end clusters compound where participants agree.
    """
    from matplotlib.lines import Line2D

    has_left  = any(len(t) >= 2 for t in left_trajectories)
    has_right = any(len(t) >= 2 for t in right_trajectories)
    if not has_left and not has_right:
        _save_empty_plot(out_path, "Cumulative Scan-Path Trajectory  (Cohort)")
        return

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=_BG_DARK)
    ax.set_facecolor(_BG_AXES)

    # Accumulate start / end coordinates separately for scatter pass
    left_starts,  left_ends  = [], []
    right_starts, right_ends = [], []

    # ── Left index paths (neon cyan) ──────────────────────────────────────────
    for raw_traj in left_trajectories:
        if len(raw_traj) < 2:
            continue
        traj = _smooth_traj(raw_traj)
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, color=_LEFT_C, alpha=0.12, linewidth=0.75,
                solid_capstyle="round", solid_joinstyle="round")
        left_starts.append((xs[0],  ys[0]))
        left_ends.append(  (xs[-1], ys[-1]))

    # ── Right index paths (coral-crimson) ─────────────────────────────────────
    for raw_traj in right_trajectories:
        if len(raw_traj) < 2:
            continue
        traj = _smooth_traj(raw_traj)
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, color=_RIGHT_C, alpha=0.12, linewidth=0.75,
                solid_capstyle="round", solid_joinstyle="round")
        right_starts.append((xs[0],  ys[0]))
        right_ends.append(  (xs[-1], ys[-1]))

    # ── START markers  (filled circle ●) ─────────────────────────────────────
    # Green for both hands so the concept "started here" reads immediately.
    _START_C = "#22ff88"
    if left_starts:
        sx, sy = zip(*left_starts)
        ax.scatter(sx, sy, s=28, color=_START_C, alpha=0.55, marker="o",
                   edgecolors="none", zorder=5)
    if right_starts:
        sx, sy = zip(*right_starts)
        ax.scatter(sx, sy, s=28, color=_START_C, alpha=0.55, marker="o",
                   edgecolors="none", zorder=5)

    # ── END markers  (cross ✕) ────────────────────────────────────────────────
    # Color-coded to match the hand: cyan for left, coral for right.
    if left_ends:
        ex, ey = zip(*left_ends)
        ax.scatter(ex, ey, s=32, color=_LEFT_C, alpha=0.55, marker="X",
                   edgecolors="none", zorder=5)
    if right_ends:
        ex, ey = zip(*right_ends)
        ax.scatter(ex, ey, s=32, color=_RIGHT_C, alpha=0.55, marker="X",
                   edgecolors="none", zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    if has_left:
        legend_handles.append(
            Line2D([0], [0], color=_LEFT_C, linewidth=2.5,
                   label=f"Left Index  (n = {len(left_trajectories)})")
        )
    if has_right:
        legend_handles.append(
            Line2D([0], [0], color=_RIGHT_C, linewidth=2.5,
                   label=f"Right Index  (n = {len(right_trajectories)})")
        )
    # Marker legend entries
    legend_handles.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_START_C,
               markersize=7, label="Start  (●)")
    )
    legend_handles.append(
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#aaaacc",
               markersize=7, label="End  (✕)  — hand-coloured")
    )
    ax.legend(handles=legend_handles, loc="upper right", facecolor="#1a1a28",
              edgecolor=_SPINE_C, labelcolor=_TEXT_C, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Cumulative Scan-Path Trajectory  (Cohort)",
                 color=_TEXT_C, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (normalised)", color=_LABEL_C, fontsize=10)
    ax.set_ylabel("Y (normalised)", color=_LABEL_C, fontsize=10)
    ax.tick_params(colors=_TICK_C)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE_C)

    n_left  = sum(1 for t in left_trajectories  if len(t) >= 2)
    n_right = sum(1 for t in right_trajectories if len(t) >= 2)
    ax.text(0.01, 0.01,
            f"Cohort: {max(n_left, n_right)} participants  "
            f"| SG-smoothed (win=11, poly=2)",
            transform=ax.transAxes, color=_LABEL_C, fontsize=7, va="bottom")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)


def generate_cumulative_heatmap(heatmap_grid, video_count, out_path):
    """
    Cohort 2-D coordinate density map rendered as smooth, organic exploration clouds.

    Pipeline:
    1. The pre-binned 30×30 raw count grid is upsampled to a 600×600 canvas
       via bilinear zoom so the subsequent Gaussian pass has enough resolution
       to produce smooth, interconnected density blobs.
    2. A 2D Gaussian filter (sigma=20 canvas pixels ≈ 1 normalised grid cell)
       is applied to blur adjacent hotspots into continuous cloud shapes that
       organically reflect where participants explored.
    3. The blurred canvas is normalised to [0, 1] and then transformed with a
       power curve (exponent 0.45) — this compresses extreme peak values so
       bright hotspots do not burn out into solid white while simultaneously
       amplifying the lower-density peripheral trails to keep them visible.
    4. Pixels whose smoothed value falls below 0.3% of peak are masked and
       rendered as the axes background colour, so completely untouched canvas
       regions remain a clean dark void with no bleed-through halo artefacts.
    5. Final render uses imshow with bicubic interpolation for silky gradients
       and the inferno colormap (dark-red → orange → bright yellow-white).
    """
    from scipy.ndimage import zoom as ndimage_zoom
    from scipy.ndimage import gaussian_filter

    if heatmap_grid.sum() == 0:
        _save_empty_plot(out_path, "Cumulative Exploration Heatmap  (Cohort)")
        return

    # ── Step 1: per-video average raw counts ──────────────────────────────────
    grid = heatmap_grid.astype(np.float64) / max(video_count, 1)

    # ── Step 2: upsample to high-resolution canvas for smooth Gaussian pass ───
    CANVAS = 600
    scale  = CANVAS // grid.shape[0]          # 600 / 30 = 20
    canvas = ndimage_zoom(grid, scale, order=1)  # bilinear upsampling

    # ── Step 3: 2D Gaussian blur — sigma in canvas pixels ────────────────────
    # sigma=20 on a 600-px canvas ≈ blending radius of ~1 normalised grid cell,
    # yielding smooth, interconnected cloud shapes over adjacent hotspot clusters.
    sigma  = 20
    canvas = gaussian_filter(canvas.astype(np.float64), sigma=sigma)

    # ── Step 4: normalise to [0, 1] then apply power compression ─────────────
    peak = float(canvas.max())
    if peak == 0:
        _save_empty_plot(out_path, "Cumulative Exploration Heatmap  (Cohort)")
        return
    canvas = canvas / peak                     # linear [0, 1]
    canvas = np.power(canvas, 0.45)            # power-curve compression

    # ── Step 5: mask true background (below 0.3% of peak after transform) ────
    # This threshold is evaluated on the power-compressed values, so low-density
    # halos that are genuine traces still pass through while empty space is masked.
    THRESHOLD = 0.003
    masked = np.ma.masked_where(canvas < THRESHOLD, canvas)

    # ── Step 6: build colormap with dark-background bad-value colour ──────────
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color=_BG_AXES)

    # ── Step 7: render ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=_BG_DARK)
    ax.set_facecolor(_BG_AXES)

    im = ax.imshow(
        masked,
        origin="upper",
        extent=[0, 1, 1, 0],
        cmap=cmap,
        vmin=THRESHOLD,
        vmax=1.0,
        aspect="auto",
        interpolation="bicubic",   # silky-smooth gradients
    )

    cbar = fig.colorbar(im, ax=ax, label="Relative Focus Intensity", pad=0.02)
    cbar.ax.yaxis.label.set_color(_LABEL_C)
    cbar.ax.tick_params(colors=_TICK_C)

    ax.set_title("Cumulative Exploration Heatmap  (Cohort)",
                 color=_TEXT_C, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (normalised)", color=_LABEL_C, fontsize=10)
    ax.set_ylabel("Y (normalised)", color=_LABEL_C, fontsize=10)
    ax.tick_params(colors=_TICK_C)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE_C)
    ax.text(0.01, 0.01,
            f"Videos: {video_count}  |  Canvas: {CANVAS}×{CANVAS}  "
            f"|  σ = {sigma}px  |  γ = 0.45",
            transform=ax.transAxes, color=_LABEL_C, fontsize=7, va="bottom")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)


def generate_cumulative_direction(direction_sum, direction_count, out_path,
                                   **_unused_kwargs):
    """
    Cohort macro-flow field rendered as a clean coarse-grid quiver plot.

    The raw 30×30 direction accumulator is downsampled to a 15×15 macro-grid
    by summing each 2×2 block of cells.  For every macro-cell the arithmetic
    mean of all accumulated (dx, dy) velocity vectors is computed, giving the
    net consensus movement direction across all participants for that region
    of the tactile graphic.

    Two noise-rejection filters are applied before any arrow is drawn:
    • COUNT THRESHOLD  — cells with fewer than 4 total frame hits are
      suppressed entirely; this low bar allows the broad shared pathways
      across the sparse tactile graphic layout to remain visible while
      single-frame noise is still removed.
    • JITTER THRESHOLD — cells whose resultant magnitude is less than 1% of
      the global peak magnitude are suppressed; these represent near-stationary
      dwell without directional intent.

    Arrow length is proportional to the mean velocity magnitude of each cell —
    fast sweeps produce long, prominent arrows; slow dwell produces short,
    subtle arrows.  The longest arrow spans 70% of its grid cell width so
    spacing between high-speed neighbours remains readable.
    Colour encodes velocity magnitude via the plasma colormap.
    """
    # ── Downsample 30×30 → 15×15 macro-grid ─────────────────────────────────
    MACRO      = 15
    fine_size  = direction_sum.shape[0]           # 30
    block      = fine_size // MACRO               # 2
    remainder  = fine_size % MACRO                # 0 if MACRO divides evenly

    # Trim to exact multiple if needed (handles edge cases gracefully)
    trim       = fine_size - remainder
    ds_trim    = direction_sum[:trim, :trim, :]
    dc_trim    = direction_count[:trim, :trim]

    # Sum 2×2 blocks: shape → (MACRO, block, MACRO, block, 2) → (MACRO, MACRO, 2)
    ds_macro   = ds_trim.reshape(MACRO, block, MACRO, block, 2).sum(axis=(1, 3))
    dc_macro   = dc_trim.reshape(MACRO, block, MACRO, block).sum(axis=(1, 3))

    # ── COUNT THRESHOLD  (suppress single-touch noise) ───────────────────────
    COUNT_MIN  = 4
    valid_mask = dc_macro >= COUNT_MIN

    # ── Average velocity per macro-cell (only where data is sufficient) ───────
    safe_dc  = np.where(valid_mask, dc_macro, 1.0)
    U_avg    = np.where(valid_mask, ds_macro[:, :, 0] / safe_dc, 0.0)
    V_avg    = np.where(valid_mask, ds_macro[:, :, 1] / safe_dc, 0.0)

    # ── JITTER THRESHOLD  (suppress near-stationary dwell cells) ─────────────
    mag_raw   = np.sqrt(U_avg ** 2 + V_avg ** 2)
    peak_mag  = float(mag_raw.max()) if mag_raw.max() > 0 else 1.0
    JITTER_T  = 0.01 * peak_mag       # 1% of peak — lenient enough for sparse data
    draw_mask = valid_mask & (mag_raw > JITTER_T)

    # ── Grid centres for quiver ───────────────────────────────────────────────
    cell_size = 1.0 / MACRO
    centres   = np.linspace(cell_size / 2.0, 1.0 - cell_size / 2.0, MACRO)
    X_m, Y_m  = np.meshgrid(centres, centres)

    # ── Scale arrows proportional to velocity magnitude ───────────────────────
    # The fastest cell gets MAX_ARROW length; all others scale linearly with their
    # own magnitude so slow dwell → short arrow, fast sweep → long arrow.
    MAX_ARROW  = 0.70 * cell_size     # 70% of cell width for the peak-velocity arrow
    scale_fac  = MAX_ARROW / max(peak_mag, 1e-12)
    U_plot     = np.where(draw_mask, U_avg * scale_fac, 0.0)
    V_plot     = np.where(draw_mask, V_avg * scale_fac, 0.0)
    mag_plot   = np.where(draw_mask, mag_raw, np.nan)

    # Flatten to 1-D arrays; drop cells that should not be drawn
    flat_mask  = draw_mask.ravel()
    Xf  = X_m.ravel()[flat_mask]
    Yf  = Y_m.ravel()[flat_mask]
    Uf  = U_plot.ravel()[flat_mask]
    Vf  = V_plot.ravel()[flat_mask]
    Cf  = mag_plot.ravel()[flat_mask]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=_BG_DARK)
    ax.set_facecolor(_BG_AXES)

    # ── Draw subtle grid lines for visual context ─────────────────────────────
    for g in centres:
        ax.axhline(g - cell_size / 2, color="#1e1e2e", linewidth=0.4,
                   zorder=1, alpha=0.6)
        ax.axvline(g - cell_size / 2, color="#1e1e2e", linewidth=0.4,
                   zorder=1, alpha=0.6)

    if len(Xf) == 0:
        ax.text(0.5, 0.5, "Insufficient tracking data\nfor direction field",
                ha="center", va="center", color=_LABEL_C, fontsize=12,
                transform=ax.transAxes)
    else:
        # quiver: scale=1 + scale_units='xy' → arrow length equals (U,V) in data coords,
        # so proportionality to velocity magnitude is preserved directly.
        q = ax.quiver(
            Xf, Yf, Uf, Vf,
            Cf,
            cmap="plasma",
            clim=(0.0, peak_mag),
            scale=1.0,
            scale_units="xy",
            angles="xy",
            width=0.010,        # thicker shaft for dark-background visibility
            headwidth=5.5,
            headlength=7.0,
            headaxislength=5.5,
            pivot="mid",
            zorder=4,
            alpha=0.92,
        )
        cbar = fig.colorbar(q, ax=ax,
                            label="Average Cohort Velocity", pad=0.02)
        cbar.ax.yaxis.label.set_color(_LABEL_C)
        cbar.ax.tick_params(colors=_TICK_C)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Cumulative Direction Flow Field  (Cohort)",
                 color=_TEXT_C, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (normalised)", color=_LABEL_C, fontsize=10)
    ax.set_ylabel("Y (normalised)", color=_LABEL_C, fontsize=10)
    ax.tick_params(colors=_TICK_C)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE_C)

    n_arrows = int(flat_mask.sum())
    ax.text(0.01, 0.01,
            f"Macro-grid: {MACRO}×{MACRO}  |  Count threshold: ≥{COUNT_MIN} frames  "
            f"|  Active cells: {n_arrows}  |  Arrow length ∝ velocity",
            transform=ax.transAxes, color=_LABEL_C, fontsize=7, va="bottom")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)


def generate_cumulative_transition_graph(transition_counts, out_path,
                                         regions=None):
    """
    Markov-chain probability graph from cohort-aggregated region transitions.

    Improvements over the previous version:
    • Bidirectional edge pairs use rad=0.28 to push opposing arcs apart;
      unidirectional edges use rad=0.12.
    • Probability labels are drawn manually with a dark-background bbox mask
      (fc=_BG_AXES) so they float beside — not on top of — the arrow lines.
    • Labels are offset perpendicular to the arc at the arc geometric midpoint
      using _arc_label_xy(), eliminating overlap with arrowheads.

    When transition_counts is empty but regions is provided, the function still
    renders all region nodes (with no edges) so the output is informative rather
    than a blank placeholder.
    """
    _TITLE = "Cumulative Region Transition Graph — Markov Probabilities  (Cohort)"

    def _nodes_only(node_names):
        """Render a nodes-only graph with a 'no transitions' annotation."""
        G_n = nx.DiGraph()
        for n in node_names:
            G_n.add_node(n)
        pos = nx.spring_layout(G_n, seed=42, k=2.6)
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=_BG_DARK)
        ax.set_facecolor(_BG_AXES)
        nx.draw_networkx_nodes(G_n, pos, ax=ax, node_color="#0ea5e9",
                               node_size=1400, alpha=0.93)
        nx.draw_networkx_labels(G_n, pos, ax=ax, font_color="#ffffff",
                                font_size=11, font_weight="bold")
        ax.text(0.5, 0.04, "No transitions detected for this hand",
                ha="center", va="bottom", transform=ax.transAxes,
                color=_LABEL_C, fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc=_BG_AXES, ec=_SPINE_C, alpha=0.9))
        ax.set_title(_TITLE, color=_TEXT_C, fontsize=12, fontweight="bold", pad=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
        plt.close(fig)

    if not transition_counts:
        if regions:
            _nodes_only(regions.keys())
        else:
            _save_empty_plot(out_path, _TITLE)
        return

    G = nx.DiGraph()
    for src, dsts in transition_counts.items():
        total = sum(dsts.values())
        for dst, count in dsts.items():
            G.add_edge(src, dst, weight=count / total, count=count)

    if not G.edges():
        node_names = list(G.nodes()) or (list(regions.keys()) if regions else [])
        if node_names:
            _nodes_only(node_names)
        else:
            _save_empty_plot(out_path, _TITLE)
        return

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w   = max(weights) if weights else 1.0

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=_BG_DARK)
    ax.set_facecolor(_BG_AXES)

    pos = nx.spring_layout(G, seed=42, k=2.6)

    # ── Classify edges as bidirectional or unidirectional ────────────────────
    bidi_edges = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    uni_edges  = [(u, v) for u, v in G.edges() if not G.has_edge(v, u)]

    RAD_UNI  = 0.12
    RAD_BIDI = 0.28

    def _draw_edge_group(edgelist, rad):
        if not edgelist:
            return
        ew = [G[u][v]["weight"] for u, v in edgelist]
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edgelist,
            edge_color=[plt.cm.plasma(w / max_w) for w in ew],
            alpha=0.85,
            arrows=True,
            arrowsize=20,
            width=[1.0 + 4.5 * w / max_w for w in ew],
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=26,
            min_target_margin=26,
        )

    _draw_edge_group(uni_edges,  RAD_UNI)
    _draw_edge_group(bidi_edges, RAD_BIDI)

    # ── Nodes & node labels ───────────────────────────────────────────────────
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#0ea5e9",
                           node_size=1400, alpha=0.93)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="#ffffff",
                            font_size=11, font_weight="bold")

    # ── Manual edge probability labels (masked bbox, perpendicular offset) ───
    for u, v in G.edges():
        rad  = RAD_BIDI if G.has_edge(v, u) else RAD_UNI
        prob = G[u][v]["weight"]
        lx, ly = _arc_label_xy(pos[u], pos[v], rad, extra=0.030)
        ax.text(lx, ly, f"{prob:.2f}",
                ha="center", va="center",
                color="#ffffff", fontsize=9, fontweight="bold",
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc=_BG_AXES, ec="none", alpha=1.0))

    ax.set_title(
        "Cumulative Region Transition Graph — Markov Probabilities  (Cohort)",
        color=_TEXT_C, fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)


def generate_cumulative_spatial_transition_graph(
        transition_counts, region_dwell_frames, regions, out_path,
        background_image_path=None,
        left_transition_counts=None,
        right_transition_counts=None):
    """
    Directed transition arrows overlaid on salient region bounding boxes.

    Pass background_image_path to superimpose the tactile graphic photo so
    spatial transitions can be understood in the context of the actual graphic.

    Improvements over the previous version:
    • Bidirectional pairs use rad=0.28; unidirectional edges use rad=0.12 —
      opposing traffic lanes no longer collide visually.
    • Probability labels use _arc_label_xy() to float beside the arc midpoint
      and are wrapped in a dark-background bbox mask to prevent overlap with
      the arrow line body.
    • Node radius scales with cumulative dwell time (size ∝ engagement).
    • Arrow width scales with Markov transition probability.
    """
    from matplotlib.lines import Line2D

    if not regions:
        _save_empty_plot(out_path, "Cumulative Spatial Transition Graph  (Cohort)")
        return

    centers = {
        name: ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        for name, b in regions.items()
        if isinstance(b, (list, tuple)) and len(b) == 4
    }

    # Row-normalised Markov probabilities
    probs = {}
    for src, dsts in transition_counts.items():
        total = sum(dsts.values())
        for dst, count in dsts.items():
            probs[(src, dst)] = count / total

    max_dwell = max(region_dwell_frames.values(), default=1)
    max_prob  = max(probs.values(),               default=1.0)

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=_BG_DARK)
    ax.set_facecolor(_BG_AXES)

    # ── Background tactile graphic image ──────────────────────────────────────
    if background_image_path and os.path.exists(str(background_image_path)):
        bg_img = cv2.imread(str(background_image_path))
        if bg_img is not None:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            # extent=[left, right, bottom, top]; ylim is inverted (0=top, 1=bottom)
            ax.imshow(bg_img, extent=[0, 1, 1, 0], aspect="auto",
                      alpha=0.40, zorder=0)

    # ── Region bounding boxes ─────────────────────────────────────────────────
    for name, b in regions.items():
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        x1, y1, x2, y2 = b
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="#0ea5e9", linewidth=1.6, zorder=2,
        ))

    # ── Region nodes scaled by dwell time ─────────────────────────────────────
    for name, (cx, cy) in centers.items():
        dwell  = region_dwell_frames.get(name, 0)
        radius = 0.014 + 0.034 * (dwell / max_dwell)
        ax.add_patch(plt.Circle(
            (cx, cy), radius, color="#0ea5e9", alpha=0.85, zorder=4,
        ))
        ax.text(cx, cy, name, ha="center", va="center",
                color="#ffffff", fontsize=9, fontweight="bold", zorder=5)

    # ── Transition arrows ─────────────────────────────────────────────────────
    # Inner helper: draw one set of transitions in a given color.
    # rad_uni / rad_bidi control arc curvature; using different values for
    # left vs right ensures same-edge arrows from both hands stay visible.
    def _draw_arrows(tc, color, rad_uni, rad_bidi):
        if not tc:
            return
        h_probs = {}
        for src, dsts in tc.items():
            total = sum(dsts.values())
            for dst, count in dsts.items():
                h_probs[(src, dst)] = count / total
        if not h_probs:
            return
        h_max = max(h_probs.values())
        for (src, dst), prob in h_probs.items():
            if src not in centers or dst not in centers:
                continue
            rad   = rad_bidi if (dst, src) in h_probs else rad_uni
            lw    = 0.9 + 5.5 * prob / h_max
            alpha = 0.32 + 0.63 * prob / h_max
            ax.annotate(
                "", xy=centers[dst], xytext=centers[src],
                arrowprops=dict(
                    arrowstyle="-|>", color=color,
                    lw=lw, alpha=alpha,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=3,
            )
            lx, ly = _arc_label_xy(centers[src], centers[dst], rad, extra=0.022)
            ax.text(lx, ly, f"{prob:.2f}",
                    ha="center", va="center",
                    color="#ffffff", fontsize=9, fontweight="bold",
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc=_BG_AXES, ec="none", alpha=1.0))

    _LEFT_ARROW  = "#ff3333"   # red   — left hand
    _RIGHT_ARROW = "#1a4fdb"   # dark blue — right hand
    _BOTH_ARROW  = "#a855f7"   # purple — single-hand or combined fallback

    two_color_mode = (left_transition_counts is not None
                      and right_transition_counts is not None)
    if two_color_mode:
        # Left hand: arcs curving slightly inward (rad +0.12/+0.28)
        # Right hand: arcs curving slightly outward (rad +0.20/+0.36) so
        # both arcs on the same edge remain visually distinct.
        _draw_arrows(left_transition_counts,  _LEFT_ARROW,  rad_uni=0.12, rad_bidi=0.28)
        _draw_arrows(right_transition_counts, _RIGHT_ARROW, rad_uni=0.20, rad_bidi=0.36)
    else:
        _draw_arrows(transition_counts, _BOTH_ARROW, rad_uni=0.12, rad_bidi=0.28)

    # ── "No transitions" annotation when the hand had no region transitions ───
    if not transition_counts:
        ax.text(0.5, 0.03, "No transitions detected for this hand",
                ha="center", va="bottom", transform=ax.transAxes,
                color=_LABEL_C, fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc=_BG_AXES, ec=_SPINE_C, alpha=0.9))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#0ea5e9",
                       label="Region  (size ∝ dwell time)"),
    ]
    if two_color_mode:
        legend_handles += [
            Line2D([0], [0], color=_LEFT_ARROW,  linewidth=2.5,
                   label="Left hand  (width ∝ probability)"),
            Line2D([0], [0], color=_RIGHT_ARROW, linewidth=2.5,
                   label="Right hand  (width ∝ probability)"),
        ]
    else:
        legend_handles.append(
            Line2D([0], [0], color=_BOTH_ARROW, linewidth=2.5,
                   label="Transition  (width ∝ probability)"),
        )
    ax.legend(handles=legend_handles, loc="upper right", facecolor="#1a1a28",
              edgecolor=_SPINE_C, labelcolor=_TEXT_C, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title("Cumulative Spatial Transition Graph  (Cohort)",
                 color=_TEXT_C, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X (normalised)", color=_LABEL_C, fontsize=10)
    ax.set_ylabel("Y (normalised)", color=_LABEL_C, fontsize=10)
    ax.tick_params(colors=_TICK_C)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE_C)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)