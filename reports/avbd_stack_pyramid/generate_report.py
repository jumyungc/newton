from __future__ import annotations

import html
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[1]
VIDEO_DIR = REPORT_DIR / "videos"
DATA_DIR = REPORT_DIR / "data"

sys.path.insert(0, str(REPO_ROOT))

import newton  # noqa: E402
from newton.examples.basic.example_basic_avbd_stack import Example  # noqa: E402

ROWS = 16
EXPECTED_SETTLED_TOP = 0.25 + (ROWS - 1) * 0.5


class DummyViewer:
    def set_model(self, model):
        pass

    def set_camera(self, *args, **kwargs):
        pass

    def apply_forces(self, state):
        pass


@dataclass
class Case:
    title: str
    slug: str
    frames: int
    contact_ke: float
    gap: float
    margin: float
    contact_kd: float
    primal_warmstart: float
    adaptive_primal_warmstart: bool
    linear_beta: float | None
    angular_beta: float | None
    contact_matching_pos_threshold: float = 0.00075
    contact_matching_normal_dot_threshold: float = 0.995
    alpha: float = 0.99
    gamma: float = 0.999
    contact_k_start: float = 1.0
    use_default_matching: bool = False
    solver_fixed_alpha_beta: bool = False
    sample_every: int = 2


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def color_for_index(i: int) -> tuple[int, int, int]:
    h = (i * 0.61803398875) % 1.0
    s = 0.58
    v = 0.86
    sector = int(h * 6.0)
    f = h * 6.0 - sector
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    vals = {
        0: (v, t, p),
        1: (q, v, p),
        2: (p, v, t),
        3: (p, q, v),
        4: (t, p, v),
        5: (v, p, q),
    }[sector % 6]
    return tuple(int(255 * c) for c in vals)


class Renderer:
    def __init__(self, width: int = 960, height: int = 540):
        self.width = width
        self.height = height
        self.target = np.array([0.0, 0.0, 4.8], dtype=np.float32)
        self.eye = np.array([10.0, -18.0, 9.0], dtype=np.float32)
        self.forward = normalize(self.target - self.eye)
        self.right = normalize(np.cross(self.forward, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
        self.up = normalize(np.cross(self.right, self.forward))
        self.scale = 34.0
        self.center = np.array([width * 0.5, height * 0.58], dtype=np.float32)
        self.light = normalize(np.array([-0.4, -0.7, 1.0], dtype=np.float32))
        self.font = ImageFont.load_default()

    def project(self, p: np.ndarray) -> tuple[float, float, float]:
        r = p - self.target
        x = float(np.dot(r, self.right))
        y = float(np.dot(r, self.up))
        z = float(np.dot(r, self.forward))
        return self.center[0] + x * self.scale, self.center[1] - y * self.scale, z

    def draw_ground(self, draw: ImageDraw.ImageDraw):
        pts = [
            np.array([-10.0, -5.0, 0.0], dtype=np.float32),
            np.array([10.0, -5.0, 0.0], dtype=np.float32),
            np.array([10.0, 5.0, 0.0], dtype=np.float32),
            np.array([-10.0, 5.0, 0.0], dtype=np.float32),
        ]
        poly = [(self.project(p)[0], self.project(p)[1]) for p in pts]
        draw.polygon(poly, fill=(38, 41, 45), outline=(70, 74, 80))

    def render(self, body_q: np.ndarray, rows: list[int], frame: int, label: str, metrics: dict) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), (18, 20, 24))
        draw = ImageDraw.Draw(image)
        self.draw_ground(draw)

        half = np.array([0.5, 0.25, 0.25], dtype=np.float32)
        corners = np.array(
            [
                [-half[0], -half[1], -half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]],
                [-half[0], half[1], -half[2]],
                [-half[0], -half[1], half[2]],
                [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]],
                [-half[0], half[1], half[2]],
            ],
            dtype=np.float32,
        )
        faces = [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]

        polys = []
        for i, q in enumerate(body_q):
            pos = q[:3].astype(np.float32)
            rot = quat_to_matrix(q[3:7].astype(np.float32))
            world = corners @ rot.T + pos
            base_color = np.array(color_for_index(i), dtype=np.float32)
            for face in faces:
                verts = world[list(face)]
                normal = normalize(np.cross(verts[1] - verts[0], verts[2] - verts[1]))
                shade = 0.52 + 0.48 * max(0.0, float(np.dot(normal, self.light)))
                color = tuple(np.clip(base_color * shade, 0, 255).astype(np.uint8).tolist())
                pts = [(self.project(p)[0], self.project(p)[1]) for p in verts]
                depth = float(np.mean([self.project(p)[2] for p in verts]))
                polys.append((depth, pts, color))

        for _depth, pts, color in sorted(polys, key=lambda item: item[0]):
            draw.polygon(pts, fill=color, outline=(12, 14, 16))

        status_color = (118, 231, 150) if metrics.get("stable", False) else (255, 111, 103)
        draw.rectangle((0, 0, self.width, 52), fill=(10, 12, 16))
        draw.text((14, 10), label, fill=(235, 238, 242), font=self.font)
        draw.text((14, 29), f"frame {frame:04d}", fill=(180, 186, 194), font=self.font)
        draw.text(
            (self.width - 350, 10),
            f"top z {metrics['top_z']:.2f} / {EXPECTED_SETTLED_TOP:.2f}",
            fill=(220, 224, 230),
            font=self.font,
        )
        draw.text(
            (self.width - 350, 29),
            f"y spread {metrics['y_abs_max']:.2f}  row delta {metrics['min_row_delta']:.2f}",
            fill=status_color,
            font=self.font,
        )
        return np.asarray(image)


def build_example(case: Case) -> Example:
    args = SimpleNamespace(
        layout="pyramid",
        contact_mode="hard_history",
        contact_ke=case.contact_ke,
        contact_kd=case.contact_kd,
        contact_k_start=case.contact_k_start,
        gap=case.gap,
        margin=case.margin,
        friction=0.5,
        rigid_primal_warmstart=case.primal_warmstart,
        rigid_adaptive_primal_warmstart=case.adaptive_primal_warmstart,
        linear_beta=case.linear_beta,
        angular_beta=case.angular_beta,
        rigid_contact_max=None,
        pyramid_rows=ROWS,
    )
    ex = Example(DummyViewer(), args)

    if case.use_default_matching:
        ex.collision_pipeline = newton.CollisionPipeline(ex.model, contact_matching="latest")
        ex.contacts = ex.model.contacts(collision_pipeline=ex.collision_pipeline)
        ex.graph = None
    elif case.contact_matching_pos_threshold != 0.00075:
        ex.collision_pipeline = newton.CollisionPipeline(
            ex.model,
            broad_phase="nxn",
            contact_matching="latest",
            contact_matching_pos_threshold=case.contact_matching_pos_threshold,
            contact_matching_normal_dot_threshold=case.contact_matching_normal_dot_threshold,
            deterministic=True,
            rigid_contact_max=max(512, 16 * len(ex.box_bodies)),
        )
        ex.contacts = ex.collision_pipeline.contacts()
        ex.graph = None

    if case.solver_fixed_alpha_beta:
        ex.solver = newton.solvers.SolverVBD(
            ex.model,
            iterations=10,
            rigid_avbd_alpha=case.alpha,
            rigid_avbd_linear_beta=0.0,
            rigid_avbd_angular_beta=0.0,
            rigid_avbd_gamma=case.gamma,
            rigid_contact_k_start=case.contact_k_start,
            rigid_contact_history=True,
            rigid_contact_hard=True,
            rigid_contact_stick_motion_eps=0.0,
            rigid_contact_stick_freeze_translation_eps=0.0,
            rigid_contact_stick_freeze_angular_eps=0.0,
            rigid_body_contact_buffer_size=512,
            rigid_primal_warmstart=case.primal_warmstart,
            rigid_adaptive_primal_warmstart=case.adaptive_primal_warmstart,
        )
        ex.graph = None

    return ex


def compute_metrics(ex: Example) -> dict:
    q = ex.state_0.body_q.numpy()[ex.box_bodies]
    qd = ex.state_0.body_qd.numpy()[ex.box_bodies]
    z = q[:, 2]
    row_ids = np.array(ex.box_rows)
    row_means = np.array([np.mean(z[row_ids == row]) for row in range(ROWS)])
    linear_speed = np.linalg.norm(qd[:, :3], axis=1)
    angular_speed = np.linalg.norm(qd[:, 3:], axis=1)
    min_row_delta = float(np.nanmin(np.diff(row_means)))
    metrics = {
        "top_z": float(np.nanmax(z)),
        "top_ratio": float(np.nanmax(z) / EXPECTED_SETTLED_TOP),
        "min_z": float(np.nanmin(z)),
        "final_linear_speed": float(np.nanmax(linear_speed)),
        "final_angular_speed": float(np.nanmax(angular_speed)),
        "xy_spread": float(np.nanmax(np.linalg.norm(q[:, :2], axis=1))),
        "y_abs_max": float(np.nanmax(np.abs(q[:, 1]))),
        "min_row_delta": min_row_delta,
        "stable": bool(
            np.isfinite(q).all()
            and np.nanmax(z) > 0.9 * EXPECTED_SETTLED_TOP
            and np.nanmax(np.abs(q[:, 1])) < 2.0
            and min_row_delta > 0.2
            and np.nanmin(z) > 0.0
        ),
    }
    return metrics


def run_case(case: Case, renderer: Renderer) -> dict:
    print(f"Running {case.slug}...")
    start = time.time()
    ex = build_example(case)
    video_path = VIDEO_DIR / f"{case.slug}.mp4"
    poster_path = VIDEO_DIR / f"{case.slug}_final.png"
    frames_written = 0
    max_contacts = 0
    max_linear_speed = 0.0
    max_angular_speed = 0.0

    with imageio.get_writer(video_path, fps=30, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in range(case.frames + 1):
            metrics = compute_metrics(ex)
            qd = ex.state_0.body_qd.numpy()[ex.box_bodies]
            max_linear_speed = max(max_linear_speed, float(np.nanmax(np.linalg.norm(qd[:, :3], axis=1))))
            max_angular_speed = max(max_angular_speed, float(np.nanmax(np.linalg.norm(qd[:, 3:], axis=1))))

            if frame % case.sample_every == 0 or frame == case.frames:
                q = ex.state_0.body_q.numpy()[ex.box_bodies]
                image = renderer.render(q, ex.box_rows, frame, case.title, metrics)
                writer.append_data(image)
                frames_written += 1
                if frame == case.frames:
                    Image.fromarray(image).save(poster_path)

            if frame < case.frames:
                ex.step()
                max_contacts = max(max_contacts, int(ex.contacts.rigid_contact_count.numpy()[0]))

    final_metrics = compute_metrics(ex)
    final_metrics["max_contacts"] = max_contacts
    final_metrics["max_linear_speed"] = max_linear_speed
    final_metrics["max_angular_speed"] = max_angular_speed
    final_metrics["elapsed_seconds"] = round(time.time() - start, 3)
    final_metrics["video"] = f"videos/{case.slug}.mp4"
    final_metrics["poster"] = f"videos/{case.slug}_final.png"
    final_metrics["frames_written"] = frames_written
    final_metrics["parameters"] = {
        "frames": case.frames,
        "frame_dt": "1/60 s",
        "sim_dt": "1/60 s",
        "sim_substeps": 1,
        "solver_iterations": 10,
        "gravity": -10,
        "body_ordering": "serial one-body-per-color rigid VBD groups",
        "contact_mode": "hard body-body contact with history",
        "contact_ke": case.contact_ke,
        "contact_kd": case.contact_kd,
        "rigid_contact_k_start": case.contact_k_start,
        "collision_gap": case.gap,
        "shape_margin": case.margin,
        "friction_mu": 0.5,
        "rigid_avbd_alpha": case.alpha,
        "rigid_avbd_linear_beta": 0.0
        if case.solver_fixed_alpha_beta
        else (10000.0 if case.linear_beta is None else case.linear_beta),
        "rigid_avbd_angular_beta": 0.0
        if case.solver_fixed_alpha_beta
        else (100.0 if case.angular_beta is None else case.angular_beta),
        "rigid_avbd_gamma": case.gamma,
        "rigid_primal_warmstart": case.primal_warmstart,
        "rigid_adaptive_primal_warmstart": case.adaptive_primal_warmstart,
        "contact_matching_pos_threshold": "default" if case.use_default_matching else case.contact_matching_pos_threshold,
        "contact_matching_normal_dot_threshold": (
            "default" if case.use_default_matching else case.contact_matching_normal_dot_threshold
        ),
        "rigid_contact_stick_motion_eps": 0.0,
        "rigid_contact_stick_freeze_translation_eps": 0.0,
        "rigid_contact_stick_freeze_angular_eps": 0.0,
        "pyramid": "16 rows, 136 boxes, full size {1, 0.5, 0.5}",
        "ground": "kinematic box, half extents (50, 50, 0.5), top at z=0",
    }
    print(f"Finished {case.slug}: stable={final_metrics['stable']} top={final_metrics['top_z']:.3f}")
    return final_metrics


def write_report(results: list[tuple[Case, dict]]) -> None:
    (DATA_DIR / "metrics.json").write_text(
        json.dumps({case.slug: metrics for case, metrics in results}, indent=2),
        encoding="utf-8",
    )

    sections = []
    for case, metrics in results:
        status = "Stable" if metrics["stable"] else "Broken"
        status_class = "stable" if metrics["stable"] else "broken"
        rows = "\n".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in metrics["parameters"].items()
        )
        metric_rows = "\n".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in metrics.items()
            if k not in {"parameters", "video", "poster"}
        )
        sections.append(
            f"""
<section>
  <h2>{html.escape(case.title)}</h2>
  <p class="{status_class}">{status}</p>
  <video controls muted loop playsinline poster="{html.escape(metrics['poster'])}">
    <source src="{html.escape(metrics['video'])}" type="video/mp4">
  </video>
  <div class="tables">
    <table><caption>Used Parameters</caption>{rows}</table>
    <table><caption>Metrics</caption>{metric_rows}</table>
  </div>
</section>
"""
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Newton AVBD Pyramid Report</title>
  <style>
    body {{
      margin: 0;
      background: #111419;
      color: #e8ebef;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1, h2 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 22px; margin-top: 34px; }}
    p {{ color: #c4cbd4; }}
    video {{
      width: 100%;
      border: 1px solid #343a44;
      background: #090b0f;
      display: block;
    }}
    .stable, .broken {{
      display: inline-block;
      margin: 0 0 12px;
      font-weight: 700;
    }}
    .stable {{ color: #76e796; }}
    .broken {{ color: #ff6f67; }}
    .tables {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #181c23;
    }}
    caption {{
      text-align: left;
      font-weight: 700;
      padding: 10px 12px;
      background: #202631;
    }}
    th, td {{
      border-top: 1px solid #2d3440;
      padding: 7px 12px;
      vertical-align: top;
    }}
    th {{
      width: 48%;
      color: #aab3c0;
      font-weight: 600;
      text-align: left;
    }}
    a {{ color: #8ab4ff; }}
  </style>
</head>
<body>
<main>
  <h1>Newton AVBD Pyramid Comparison</h1>
  <p>
    Full 16-row avbd-demo3d pyramid, same computational budget as the reference:
    60 Hz, one substep, ten VBD iterations, gravity -10. The videos compare
    forward warm-start choices and the softer parameter set that looked partially
    stable during debugging.
  </p>
  <p>
    Stability criterion here is intentionally mechanical: finite poses, top box center
    above 90% of the compact stacked height ({EXPECTED_SETTLED_TOP:.2f}), row means still
    ordered upward, no body below the ground, and max out-of-plane |y| below 2 m.
  </p>
  {''.join(sections)}
</main>
</body>
</html>
"""
    (REPORT_DIR / "stack_pyramid_report.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wp.init()
    if wp.is_cuda_available():
        wp.set_device("cuda:0")

    cases = [
        Case(
            title="Stiff Hard History, Adaptive Reference Warm Start",
            slug="stiff_hard_history_adaptive_warmstart",
            frames=240,
            contact_ke=1.0e8,
            gap=0.0,
            margin=0.01,
            contact_kd=0.0,
            primal_warmstart=0.0,
            adaptive_primal_warmstart=True,
            linear_beta=None,
            angular_beta=None,
        ),
        Case(
            title="Stiff Hard History, Original Forward Step",
            slug="stiff_hard_history_original_forward",
            frames=240,
            contact_ke=1.0e8,
            gap=0.0,
            margin=0.01,
            contact_kd=0.0,
            primal_warmstart=1.0,
            adaptive_primal_warmstart=False,
            linear_beta=None,
            angular_beta=None,
        ),
        Case(
            title="Soft Debug Setup, ke=100 and gap=0.1",
            slug="soft_debug_ke100_gap01",
            frames=480,
            contact_ke=1.0e2,
            gap=0.1,
            margin=0.0,
            contact_kd=1.0e-5,
            primal_warmstart=0.0,
            adaptive_primal_warmstart=True,
            linear_beta=0.0,
            angular_beta=0.0,
            alpha=0.95,
            use_default_matching=True,
            solver_fixed_alpha_beta=True,
        ),
    ]

    renderer = Renderer()
    results = [(case, run_case(case, renderer)) for case in cases]
    write_report(results)
    print(f"Wrote {REPORT_DIR / 'stack_pyramid_report.html'}")


if __name__ == "__main__":
    main()
