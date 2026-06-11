"""
3-D visualisation of ``ee_quat_pos`` action vectors.

Renders ground-truth and predicted end-effector poses as coloured arrows
on a Matplotlib 3-D axis.  Each 16-element vector is split into two arms:

    [x, y, z, qi, qj, qk, qw, gripper] × 2   (left arm | right arm)

Gripper state is ignored.  Quaternion ``(qi, qj, qk, qw)`` is converted
to a rotation matrix; the local **+x** axis is drawn as the arrow.

Colours:
    * **green** — ground-truth
    * **red**   — prediction

The main entry point is :func:`render_eef_frame` which returns an RGB
``np.ndarray`` suitable for stacking into a video.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend – no display needed

import io

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers projection

# ------------------------------------------------------------------
# quaternion → rotation matrix  (scalar-last: qi, qj, qk, qw)
# ------------------------------------------------------------------


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion ``[qi, qj, qk, qw]`` to a 3×3 rotation matrix."""
    qi, qj, qk, qw = q[0], q[1], q[2], q[3]
    # normalise just in case
    n = np.sqrt(qi * qi + qj * qj + qk * qk + qw * qw) + 1e-12
    qi, qj, qk, qw = qi / n, qj / n, qk / n, qw / n

    return np.array(
        [
            [1 - 2 * (qj * qj + qk * qk), 2 * (qi * qj - qk * qw), 2 * (qi * qk + qj * qw)],
            [2 * (qi * qj + qk * qw), 1 - 2 * (qi * qi + qk * qk), 2 * (qj * qk - qi * qw)],
            [2 * (qi * qk - qj * qw), 2 * (qj * qk + qi * qw), 1 - 2 * (qi * qi + qj * qj)],
        ]
    )


# ------------------------------------------------------------------
# parse 16-d vector → two (pos, rotmat) pairs
# ------------------------------------------------------------------


def _parse_eef_quat_pos(vec: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(pos_left, rot_left), (pos_right, rot_right)]`` from a 16-d vector."""
    vec = np.asarray(vec, dtype=np.float64).ravel()
    assert vec.shape[0] == 16, f"Expected 16 elements, got {vec.shape[0]}"

    arms: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in (0, 8):
        pos = vec[offset : offset + 3]
        quat = vec[offset + 3 : offset + 7]  # qi, qj, qk, qw
        rot = _quat_to_rotmat(quat)
        arms.append((pos, rot))
    return arms


# ------------------------------------------------------------------
# drawing helpers
# ------------------------------------------------------------------

_ARROW_LEN = 0.04  # length of orientation arrows (in world units)


def _draw_arm(
    ax: Axes3D,
    pos: np.ndarray,
    rot: np.ndarray,
    colour: str,
    label: str | None = None,
    arrow_length: float = _ARROW_LEN,
) -> None:
    """Draw a position dot + three orientation arrows for one arm."""
    ax.scatter(*pos, color=colour, s=40, depthshade=True, label=label)

    # Draw the local x (forward), y (left), z (up) axes
    axis_colours = [colour, colour, colour]
    axis_alphas = [1.0, 0.4, 0.4]
    axis_widths = [2.0, 1.0, 1.0]
    for i in range(3):
        direction = rot[:, i] * arrow_length
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            direction[0],
            direction[1],
            direction[2],
            color=axis_colours[i],
            alpha=axis_alphas[i],
            linewidth=axis_widths[i],
            arrow_length_ratio=0.25,
        )


def _auto_limits(
    all_positions: Sequence[np.ndarray],
    pad: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute axis limits that comfortably contain all positions."""
    pts = np.stack(all_positions)
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad

    # enforce equal aspect by expanding to a cube
    centre = (lo + hi) / 2
    half = max((hi - lo).max() / 2, 0.01)
    lo = centre - half
    hi = centre + half
    return (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])


# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------


def render_eef_frame(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    title: str = "",
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 100,
    elev: float = 25.0,
    azim: float = -60.0,
    arrow_length: float = _ARROW_LEN,
) -> np.ndarray:
    """Render a single frame comparing GT vs predicted ``ee_quat_pos``.

    Args:
        gt:   Ground-truth 16-d ``ee_quat_pos`` vector.
        pred: Predicted 16-d ``ee_quat_pos`` vector.
        title: Optional title string.
        figsize: Matplotlib figure size in inches.
        dpi: Dots per inch for rasterisation.
        elev: Elevation angle for 3-D view.
        azim: Azimuth angle for 3-D view.
        arrow_length: Length of orientation arrows in world units.

    Returns:
        ``(H, W, 3)`` uint8 RGB image as a NumPy array.
    """
    gt_arms = _parse_eef_quat_pos(gt)
    pred_arms = _parse_eef_quat_pos(pred)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    arm_labels = ["Left", "Right"]
    all_positions: list[np.ndarray] = []

    for i, (arm_label, (gt_pos, gt_rot), (pr_pos, pr_rot)) in enumerate(
        zip(arm_labels, gt_arms, pred_arms, strict=False)
    ):
        gt_label = f"GT {arm_label}" if i == 0 else None
        pr_label = f"Pred {arm_label}" if i == 0 else None

        _draw_arm(ax, gt_pos, gt_rot, "green", label=gt_label, arrow_length=arrow_length)
        _draw_arm(ax, pr_pos, pr_rot, "red", label=pr_label, arrow_length=arrow_length)

        # thin dashed line connecting GT ↔ pred for same arm
        ax.plot(
            [gt_pos[0], pr_pos[0]],
            [gt_pos[1], pr_pos[1]],
            [gt_pos[2], pr_pos[2]],
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

        all_positions.extend([gt_pos, pr_pos])

    # axis limits
    xlim, ylim, zlim = _auto_limits(all_positions)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout()

    # rasterise to RGB array
    buf = io.BytesIO()
    fig.savefig(buf, format="raw", dpi=dpi)
    buf.seek(0)
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)

    return img[:, :, :3]  # drop alpha


def render_eef_trajectory(
    gt_seq: np.ndarray,
    pred_seq: np.ndarray,
    *,
    fps: int = 10,
    title_prefix: str = "Step",
    **kwargs,
) -> list[np.ndarray]:
    """Render a sequence of frames for an action chunk.

    Args:
        gt_seq:   ``(T, 16)`` ground-truth action chunk.
        pred_seq: ``(T, 16)`` predicted action chunk.
        fps: Not used directly but documented for callers that write video.
        title_prefix: Prefix for per-frame titles.
        **kwargs: Forwarded to :func:`render_eef_frame`.

    Returns:
        List of ``(H, W, 3)`` uint8 frames.
    """
    gt_seq = np.asarray(gt_seq, dtype=np.float64)
    pred_seq = np.asarray(pred_seq, dtype=np.float64)
    assert gt_seq.shape == pred_seq.shape, f"Shape mismatch: gt {gt_seq.shape} vs pred {pred_seq.shape}"
    n_steps = gt_seq.shape[0]
    frames: list[np.ndarray] = []
    for t in range(n_steps):
        title = f"{title_prefix} {t}/{n_steps - 1}"
        frame = render_eef_frame(gt_seq[t], pred_seq[t], title=title, **kwargs)
        frames.append(frame)
    return frames
