# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable NV72 Tray
#
# Builds a schematic NV72 tray cable bundle from the wafer mapping table and
# turns close-up bracket cable arrays into Newton rod/cable bodies solved with
# VBD. This standalone example folds the route-generation helpers into the
# same file so it can be run directly from the examples browser.
#
# Command:
#   uv run -m newton.examples cable_nv72_tray
#
###########################################################################

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TextIO

import numpy as np
import warp as wp

import newton
import newton.examples


def _np_transform_to_wp(row: np.ndarray) -> wp.transform:
    return wp.transform(
        wp.vec3(float(row[0]), float(row[1]), float(row[2])),
        wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
    )


EXPECTED_HEADER = [
    "Port",
    "L-1",
    "L-2",
    "L-3",
    "L-4",
    "L-5",
    "L-6",
    "L-7",
    "L-8",
    "L-9",
    "L-10",
    "Guide",
    "R-1",
    "R-2",
    "R-3",
    "R-4",
    "R-5",
    "R-6",
    "R-7",
    "R-8",
    "R-9",
]
USABLE_COLUMNS = [f"L-{i}" for i in range(2, 11)] + [f"R-{i}" for i in range(1, 10)]
CP_PORTS = {f"CP{i}" for i in range(1, 19)}
SW_PORTS = {f"SW{i}" for i in range(1, 19)}
ALL_PORTS = CP_PORTS | SW_PORTS
CABLES_PER_CONNECTION = 8
CABLE_THICKNESS_M = 1.4e-3
CABLE_WIDTH_M = 3.0e-3
BRACKET_ARRAY_ROWS = 8
BRACKET_ARRAY_COLUMNS = 12


_PORT_RE = re.compile(r"^(CP|SW)([1-9]|1[0-8])$")
_ENDPOINT_RE = re.compile(r"^(CP|SW)([1-9]|1[0-8])-(L|R)-([1-9]|10)$")


@dataclass(frozen=True)
class WaferEndpoint:
    """A logical port-side wafer position."""

    port: str
    side: str
    position: int

    @property
    def label(self) -> str:
        return f"{self.port}-{self.side}-{self.position}"


@dataclass(frozen=True)
class WaferConnection:
    """One CSV-listed wafer-to-wafer connection."""

    source: WaferEndpoint
    target: WaferEndpoint

    @property
    def label(self) -> str:
        return f"{self.source.label}__{self.target.label}"


@dataclass(frozen=True)
class CableInstance:
    """One physical cable within an eight-cable wafer connection."""

    connection: WaferConnection
    cable_index: int

    @property
    def label(self) -> str:
        return f"{self.connection.label}__c{self.cable_index:02d}"


@dataclass(frozen=True)
class Netlist:
    """Parsed production topology."""

    source_path: str
    source_sha256: str
    row_ports: tuple[str, ...]
    connections: tuple[WaferConnection, ...]

    def expand_cables(self, cables_per_connection: int = CABLES_PER_CONNECTION) -> list[CableInstance]:
        return [
            CableInstance(connection=connection, cable_index=i)
            for connection in self.connections
            for i in range(cables_per_connection)
        ]


@dataclass(frozen=True)
class TrayGeometry:
    """Inferred schematic mechanical layout [m].

    The real tray photo shows a front-edge connector comb feeding a shallow
    cable bed. This seed model uses:

    - X along the ordered connector row.
    - Y from the connector row into the cable bed/routing lanes.
    - Z as vertical stack height, with R wafer endpoints above L endpoints.

    These are still schematic dimensions until measured tray CAD is available.
    """

    rack_width: float = 0.600
    rack_depth: float = 1.068
    rack_height: float = 2.495
    margin_x: float = 0.030
    connector_y: float = 0.0
    lead_in_y: float = -0.055
    lane_pitch: float = CABLE_WIDTH_M
    layer_pitch: float = 2.0e-3
    layer_floor_z: float = 0.015
    l_wafer_z: float = 0.026
    r_wafer_z: float = 0.036
    wafer_position_pitch_y: float = CABLE_WIDTH_M
    min_bend_radius: float = 0.030
    route_segment_length: float = 0.020

    @property
    def usable_width(self) -> float:
        return self.rack_width - 2.0 * self.margin_x

    @property
    def usable_depth(self) -> float:
        return max(0.0, self.rack_depth - abs(self.lead_in_y) - 0.020)

    @property
    def lane_count(self) -> int:
        return max(1, int(self.usable_depth / self.lane_pitch))

    @property
    def cable_radius(self) -> float:
        return 0.5 * CABLE_THICKNESS_M


@dataclass(frozen=True)
class BracketArrayGeometry:
    """Close-up connector bracket layout [m].

    The close-up tray photo shows each bracket carrying an approximately 8 by
    12 array of individual 1.4 mm cables.  Cables leave the bracket vertically
    before bending toward a receiving plug.
    """

    rows: int = BRACKET_ARRAY_ROWS
    columns: int = BRACKET_ARRAY_COLUMNS
    cable_diameter: float = CABLE_THICKNESS_M
    cable_pitch: float = 2.0e-3
    bracket_top_z: float = 0.012
    bracket_height: float = 0.012
    vertical_lead: float = 0.050
    arch_height: float = 0.055
    plug_separation_x: float = 0.0
    receiver_offset_y: float = 0.145
    route_segment_length: float = 0.005
    bracket_margin: float = 0.006

    @property
    def cable_radius(self) -> float:
        return 0.5 * self.cable_diameter

    @property
    def cable_count(self) -> int:
        return self.rows * self.columns

    @property
    def array_width(self) -> float:
        return (self.columns - 1) * self.cable_pitch

    @property
    def array_depth(self) -> float:
        return (self.rows - 1) * self.cable_pitch


@dataclass(frozen=True)
class CableRoute:
    """A seeded cable route."""

    cable: CableInstance
    layer: int
    slot: int
    points: tuple[tuple[float, float, float], ...]
    bend_radii: tuple[float | None, ...]
    length: float

    @property
    def label(self) -> str:
        return self.cable.label


def generate_bracket_array_routes(
    geometry: BracketArrayGeometry | None = None,
    *,
    cable_limit: int | None = None,
) -> tuple[list[CableRoute], dict[str, float | int | str]]:
    """Generate the close-up bracket array shown in the tray photo."""

    geometry = geometry or BracketArrayGeometry()
    limit = geometry.cable_count if cable_limit is None else min(cable_limit, geometry.cable_count)

    routes: list[CableRoute] = []
    for route_index in range(limit):
        row = route_index // geometry.columns
        column = route_index % geometry.columns
        source = WaferEndpoint(port="SRC", side="B", position=route_index)
        target = WaferEndpoint(port="RCV", side="B", position=route_index)
        cable = CableInstance(connection=WaferConnection(source=source, target=target), cable_index=route_index)
        points = tuple(
            _deduplicate_points(
                resample_polyline(_bracket_array_route_points(row, column, geometry), geometry.route_segment_length)
            )
        )
        radii = tuple(compute_bend_radii(points))
        routes.append(
            CableRoute(
                cable=cable,
                layer=row,
                slot=column,
                points=points,
                bend_radii=radii,
                length=polyline_length(points),
            )
        )

    segment_lengths = [_distance(a, b) for route in routes for a, b in pairwise(route.points)]
    finite_radii = [r for route in routes for r in route.bend_radii if r is not None and math.isfinite(r)]
    metrics: dict[str, float | int | str] = {
        "source_path": "bracket-array",
        "source_sha256": "not-applicable",
        "order_strategy": "bracket-array",
        "order_seed": 0,
        "connection_count": len(routes),
        "cable_count": len(routes),
        "full_cable_count": geometry.cable_count,
        "cables_per_connection": 1,
        "lane_count": geometry.columns,
        "num_layers": geometry.rows,
        "max_stack_height_m": geometry.array_depth,
        "total_cable_length_m": sum(route.length for route in routes),
        "max_cable_length_m": max((route.length for route in routes), default=0.0),
        "max_segment_length_m": max(segment_lengths, default=0.0),
        "mean_segment_length_m": sum(segment_lengths) / max(1, len(segment_lengths)),
        "worst_bend_radius_m": min(finite_radii) if finite_radii else math.inf,
        "bracket_rows": geometry.rows,
        "bracket_columns": geometry.columns,
        "vertical_lead_m": geometry.vertical_lead,
        "plug_separation_x_m": geometry.plug_separation_x,
        "plug_offset_y_m": geometry.receiver_offset_y,
        "route_segment_length_m": geometry.route_segment_length,
        "centerline_profile": "c1-smooth-vertical-endpoint-arch",
        "cable_thickness_m": geometry.cable_diameter,
        "cable_width_m": geometry.cable_pitch,
    }
    return routes, metrics


def default_port_order_from_image() -> list[str]:
    """Return the port order shown in the requirements diagram."""

    return (
        [f"CP{i}" for i in range(18, 9, -1)]
        + [f"SW{i}" for i in range(18, 9, -1)]
        + [f"SW{i}" for i in range(9, 0, -1)]
        + [f"CP{i}" for i in range(9, 0, -1)]
    )


def create_demo_netlist_csv() -> str:
    """Create a deterministic reciprocal NV72-shaped mapping CSV.

    This is only for standalone examples/tests. Production runs should pass
    the real ``netlist-mapping.csv`` through :func:`load_netlist_csv`.
    """

    port_order = default_port_order_from_image()
    cp_ports = [p for p in port_order if p.startswith("CP")]
    sw_ports = [p for p in port_order if p.startswith("SW")]
    cp_endpoints = [
        WaferEndpoint(port=port, side=column[0], position=int(column.split("-")[1]))
        for port in cp_ports
        for column in USABLE_COLUMNS
    ]
    sw_endpoints = [
        WaferEndpoint(port=port, side=column[0], position=int(column.split("-")[1]))
        for port in sw_ports
        for column in USABLE_COLUMNS
    ]
    cp_to_sw = dict(zip((e.label for e in cp_endpoints), (e.label for e in sw_endpoints), strict=True))
    sw_to_cp = {target: source for source, target in cp_to_sw.items()}

    rows = [",".join(EXPECTED_HEADER)]
    for port in port_order:
        values = {"Port": port, "L-1": "NC", "Guide": "GUIDE"}
        for column in USABLE_COLUMNS:
            endpoint = parse_column_endpoint(port, column)
            values[column] = cp_to_sw[endpoint.label] if port.startswith("CP") else sw_to_cp[endpoint.label]
        rows.append(",".join(values[column] for column in EXPECTED_HEADER))
    return "\n".join(rows) + "\n"


def parse_endpoint_label(label: str) -> WaferEndpoint:
    match = _ENDPOINT_RE.match(label.strip())
    if match is None:
        raise ValueError(f"Malformed endpoint label: {label!r}")

    kind, number, side, position = match.groups()
    port = f"{kind}{number}"
    pos = int(position)
    if side == "L" and not (1 <= pos <= 10):
        raise ValueError(f"Left endpoint position out of range: {label!r}")
    if side == "R" and not (1 <= pos <= 9):
        raise ValueError(f"Right endpoint position out of range: {label!r}")
    return WaferEndpoint(port=port, side=side, position=pos)


def parse_column_endpoint(port: str, column: str) -> WaferEndpoint:
    side, position = column.split("-", maxsplit=1)
    return WaferEndpoint(port=port, side=side, position=int(position))


def _read_csv_text(path_or_file: str | Path | TextIO) -> tuple[str, str]:
    if hasattr(path_or_file, "read"):
        source_path = getattr(path_or_file, "name", "<memory>")
        text = path_or_file.read()
    else:
        path = Path(path_or_file)
        source_path = str(path)
        text = path.read_text(encoding="utf-8")
    return source_path, text


def load_netlist_csv(path_or_file: str | Path | TextIO) -> Netlist:
    """Load and validate a production mapping CSV.

    Args:
        path_or_file: CSV path or text file object.

    Returns:
        Parsed netlist with 324 CP-to-SW wafer connections.

    Raises:
        ValueError: If table shape, endpoint names, reciprocity, or cardinality
            is invalid.
    """

    source_path, text = _read_csv_text(path_or_file)
    source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    reader = csv.DictReader(text.splitlines())
    errors: list[str] = []

    if reader.fieldnames != EXPECTED_HEADER:
        errors.append(f"CSV header mismatch: expected {EXPECTED_HEADER}, got {reader.fieldnames}")

    row_ports: list[str] = []
    all_mappings: dict[str, str] = {}
    connections: list[WaferConnection] = []

    for row_index, row in enumerate(reader, start=2):
        port = (row.get("Port") or "").strip()
        if _PORT_RE.match(port) is None:
            errors.append(f"row {row_index}: invalid port name {port!r}")
            continue
        if port in row_ports:
            errors.append(f"row {row_index}: duplicate port row {port}")
        row_ports.append(port)

        if (row.get("L-1") or "").strip() != "NC":
            errors.append(f"row {row_index} {port}: L-1 must be NC")
        if (row.get("Guide") or "").strip() != "GUIDE":
            errors.append(f"row {row_index} {port}: Guide must be GUIDE")

        for column in USABLE_COLUMNS:
            source = parse_column_endpoint(port, column)
            target_label = (row.get(column) or "").strip()
            try:
                target = parse_endpoint_label(target_label)
            except ValueError as exc:
                errors.append(f"row {row_index} {port} {column}: {exc}")
                continue

            if target.port not in ALL_PORTS:
                errors.append(f"row {row_index} {port} {column}: target port out of inventory {target.port}")
            if port.startswith("CP") and not target.port.startswith("SW"):
                errors.append(f"row {row_index} {port} {column}: CP rows must target SW endpoints")
            if port.startswith("SW") and not target.port.startswith("CP"):
                errors.append(f"row {row_index} {port} {column}: SW rows must target CP endpoints")

            all_mappings[source.label] = target.label
            if port.startswith("CP"):
                connections.append(WaferConnection(source=source, target=target))

    cp_rows = [p for p in row_ports if p.startswith("CP")]
    sw_rows = [p for p in row_ports if p.startswith("SW")]
    if len(cp_rows) != 18:
        errors.append(f"expected 18 CP rows, got {len(cp_rows)}")
    if len(sw_rows) != 18:
        errors.append(f"expected 18 SW rows, got {len(sw_rows)}")
    if len(connections) != 324:
        errors.append(f"expected 324 CP-to-SW connections, got {len(connections)}")

    source_labels = [c.source.label for c in connections]
    target_labels = [c.target.label for c in connections]
    if len(set(source_labels)) != len(source_labels):
        errors.append("duplicate CP source endpoints in mapping")
    if len(set(target_labels)) != len(target_labels):
        errors.append("duplicate SW target endpoints in mapping")

    for connection in connections:
        reciprocal = all_mappings.get(connection.target.label)
        if reciprocal != connection.source.label:
            errors.append(
                f"non-reciprocal mapping: {connection.source.label} -> {connection.target.label}, "
                f"but reverse is {reciprocal!r}"
            )

    if errors:
        raise ValueError("Invalid NV72 netlist CSV:\n" + "\n".join(f"- {error}" for error in errors))

    return Netlist(
        source_path=source_path,
        source_sha256=source_sha256,
        row_ports=tuple(row_ports),
        connections=tuple(connections),
    )


def endpoint_position(
    endpoint: WaferEndpoint,
    geometry: TrayGeometry,
    port_order: list[str],
    cable_index: int = 0,
) -> tuple[float, float, float]:
    """Map a wafer endpoint to inferred schematic coordinates [m]."""

    try:
        port_idx = port_order.index(endpoint.port)
    except ValueError:
        port_order = default_port_order_from_image()
        port_idx = port_order.index(endpoint.port)

    if len(port_order) == 1:
        x = 0.5 * geometry.rack_width
    else:
        pitch_x = geometry.usable_width / float(len(port_order) - 1)
        x = geometry.margin_x + float(port_idx) * pitch_x

    if endpoint.side == "L":
        center_pos = 6.0
        z = geometry.l_wafer_z
    else:
        center_pos = 5.0
        z = geometry.r_wafer_z

    # Preserve bundle order while fanning from wafer endpoint to tray lane.  The
    # input order places cable 0 before cable 7; lanes advance in the negative Y
    # direction, so endpoint offsets must do the same to avoid a same-bundle
    # braid in the initial seed.
    cable_offset = (0.5 * float(CABLES_PER_CONNECTION - 1) - float(cable_index)) * geometry.lane_pitch
    y = geometry.connector_y + (float(endpoint.position) - center_pos) * geometry.wafer_position_pitch_y
    return (x, y + cable_offset, z)


def order_cables(
    cables: list[CableInstance],
    geometry: TrayGeometry,
    port_order: list[str],
    strategy: str,
    seed: int = 0,
) -> list[CableInstance]:
    """Apply an explicit, reproducible placement order strategy."""

    if strategy == "input":
        return list(cables)
    if strategy == "random":
        ordered = list(cables)
        rng = random.Random(seed)
        rng.shuffle(ordered)
        return ordered

    def span(cable: CableInstance) -> float:
        a = endpoint_position(cable.connection.source, geometry, port_order, cable.cable_index)
        b = endpoint_position(cable.connection.target, geometry, port_order, cable.cable_index)
        return _distance(a, b)

    if strategy == "shortest-first":
        return sorted(cables, key=span)
    if strategy == "longest-first":
        return sorted(cables, key=span, reverse=True)
    if strategy == "grouped-by-port":
        return sorted(
            cables,
            key=lambda c: (
                c.connection.source.port,
                c.connection.target.port,
                c.connection.source.side,
                c.connection.source.position,
                c.cable_index,
            ),
        )
    if strategy == "grouped-by-wafer":
        return sorted(
            cables,
            key=lambda c: (
                c.connection.source.side,
                c.connection.source.position,
                c.connection.target.side,
                c.connection.target.position,
                c.connection.source.port,
                c.cable_index,
            ),
        )

    raise ValueError(f"Unknown order strategy {strategy!r}")


def generate_routes(
    netlist: Netlist,
    geometry: TrayGeometry,
    *,
    strategy: str = "input",
    seed: int = 0,
    cable_limit: int | None = None,
) -> tuple[list[CableRoute], dict[str, float | int | str]]:
    """Generate seed routes from a netlist and inferred tray geometry."""

    port_order = [p for p in netlist.row_ports if p in ALL_PORTS]
    if len(port_order) != 36:
        port_order = default_port_order_from_image()

    cables = netlist.expand_cables()
    ordered = order_cables(cables, geometry, port_order, strategy=strategy, seed=seed)
    if cable_limit is not None:
        ordered = ordered[:cable_limit]

    routes: list[CableRoute] = []
    for order_idx, cable in enumerate(ordered):
        slot = order_idx % geometry.lane_count
        layer = order_idx // geometry.lane_count
        lane_y = geometry.lead_in_y - float(slot) * geometry.lane_pitch
        stack_z = geometry.layer_floor_z + geometry.cable_radius + float(layer) * geometry.layer_pitch
        points = _route_points(cable, geometry, port_order, lane_y, stack_z)
        points = tuple(_deduplicate_points(resample_polyline(points, geometry.route_segment_length)))
        radii = tuple(compute_bend_radii(points))
        routes.append(
            CableRoute(
                cable=cable,
                layer=layer,
                slot=slot,
                points=points,
                bend_radii=radii,
                length=polyline_length(points),
            )
        )

    finite_radii = [r for route in routes for r in route.bend_radii if r is not None and math.isfinite(r)]
    metrics: dict[str, float | int | str] = {
        "source_path": netlist.source_path,
        "source_sha256": netlist.source_sha256,
        "order_strategy": strategy,
        "order_seed": seed,
        "connection_count": len(netlist.connections),
        "cable_count": len(routes),
        "full_cable_count": len(cables),
        "cables_per_connection": CABLES_PER_CONNECTION,
        "lane_count": geometry.lane_count,
        "num_layers": (max((r.layer for r in routes), default=-1) + 1),
        "max_stack_height_m": (max((r.layer for r in routes), default=-1) + 1) * geometry.layer_pitch,
        "total_cable_length_m": sum(route.length for route in routes),
        "max_cable_length_m": max((route.length for route in routes), default=0.0),
        "worst_bend_radius_m": min(finite_radii) if finite_radii else math.inf,
        "rack_width_m": geometry.rack_width,
        "rack_depth_m": geometry.rack_depth,
        "rack_height_m": geometry.rack_height,
        "cable_thickness_m": CABLE_THICKNESS_M,
        "cable_width_m": CABLE_WIDTH_M,
    }
    return routes, metrics


def write_routes_json(path: str | Path, routes: list[CableRoute], metrics: dict[str, float | int | str]) -> None:
    """Write route geometry and metrics as JSON."""

    payload = {
        "metrics": metrics,
        "cables": [
            {
                "id": route.label,
                "source": route.cable.connection.source.label,
                "target": route.cable.connection.target.label,
                "cable_index": route.cable.cable_index,
                "layer": route.layer,
                "slot": route.slot,
                "length_m": route.length,
                "points_m": route.points,
                "bend_radii_m": route.bend_radii,
            }
            for route in routes
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_route_clearance(
    routes: list[CableRoute],
    *,
    cable_radius: float = 0.5 * CABLE_THICKNESS_M,
) -> dict[str, float | int | str | None]:
    """Measure centerline clearance for seeded routes.

    Adjacent and next-neighbor segments within a single route are ignored
    because they are part of the same continuous rod bend.  More distant
    same-route checks catch self-crossings; different-route checks catch
    initial inter-cable collisions before VBD relaxation.
    """

    segments: list[tuple[int, int, tuple[float, float, float], tuple[float, float, float]]] = []
    for route_idx, route in enumerate(routes):
        for seg_idx, (a, b) in enumerate(pairwise(route.points)):
            segments.append((route_idx, seg_idx, a, b))

    min_distance = math.inf
    min_self_distance = math.inf
    min_inter_distance = math.inf
    closest: tuple[int, int, int, int] | None = None
    closest_self: tuple[int, int, int, int] | None = None
    closest_inter: tuple[int, int, int, int] | None = None

    for i, (route_i, seg_i, a0, a1) in enumerate(segments):
        for route_j, seg_j, b0, b1 in segments[i + 1 :]:
            same_route = route_i == route_j
            if same_route and abs(seg_i - seg_j) <= 2:
                continue
            dist = segment_distance(a0, a1, b0, b1)
            if dist < min_distance:
                min_distance = dist
                closest = (route_i, seg_i, route_j, seg_j)
            if same_route and dist < min_self_distance:
                min_self_distance = dist
                closest_self = (route_i, seg_i, route_j, seg_j)
            if not same_route and dist < min_inter_distance:
                min_inter_distance = dist
                closest_inter = (route_i, seg_i, route_j, seg_j)

    diameter = 2.0 * cable_radius
    return {
        "min_centerline_distance_m": min_distance,
        "min_clearance_m": min_distance - diameter,
        "min_self_centerline_distance_m": min_self_distance,
        "min_self_clearance_m": min_self_distance - diameter,
        "min_inter_centerline_distance_m": min_inter_distance,
        "min_inter_clearance_m": min_inter_distance - diameter,
        "closest_pair": _format_segment_pair(closest),
        "closest_self_pair": _format_segment_pair(closest_self),
        "closest_inter_pair": _format_segment_pair(closest_inter),
    }


def build_vbd_model(
    routes: list[CableRoute],
    *,
    cable_radius: float = 0.5 * CABLE_THICKNESS_M,
    bracket_geometry: BracketArrayGeometry | None = None,
    bend_stiffness: float = 1.0e-2,
    bend_damping: float = 1.0e-4,
    stretch_stiffness: float = 1.0e4,
    stretch_damping: float = 1.0e-5,
) -> tuple[newton.Model, list[list[int]]]:
    """Build a Newton VBD-ready rod model from routed cable centerlines."""

    builder = newton.ModelBuilder()
    builder.rigid_gap = max(1.0e-4, 0.25 * cable_radius)
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e-2
    builder.default_shape_cfg.mu = 1.5
    builder.default_shape_cfg.density = 1.0e6
    builder.add_ground_plane()
    if bracket_geometry is not None:
        _add_bracket_array_visuals(builder, bracket_geometry)

    route_bodies: list[list[int]] = []
    for route in routes:
        points_wp = [wp.vec3(*point) for point in route.points]
        quats = newton.utils.create_parallel_transport_cable_quaternions(points_wp)
        shape_start = len(builder.shape_color)
        bodies, _joints = builder.add_rod(
            positions=points_wp,
            quaternions=quats,
            radius=cable_radius,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            label=route.label,
        )
        for shape_index in range(shape_start, len(builder.shape_color)):
            builder.shape_color[shape_index] = (0.86, 0.46, 0.26)
        # Pin the connector lead-in capsules.  Final CAD should replace this
        # with explicit connector anchor geometry.
        if bodies:
            builder.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)
            builder.body_flags[bodies[-1]] = int(newton.BodyFlags.KINEMATIC)

        # Filter near-neighbor self-collision along the cable.  Adjacent (i, i+1)
        # capsules are already filtered by the cable joint; this adds i+2 to avoid
        # spurious self-contact from capsule-cap overlap at bends, while leaving
        # distant pairs collidable so genuine coiling/fold-back still collides.
        self_filter_window = 2
        route_shapes = [builder.body_shapes[b][0] for b in bodies if builder.body_shapes.get(b)]
        for a in range(len(route_shapes)):
            for b in range(a + 2, min(a + 1 + self_filter_window, len(route_shapes))):
                builder.add_shape_collision_filter_pair(route_shapes[a], route_shapes[b])

        route_bodies.append(bodies)

    # Filter shape pairs that are already penetrating in the initial configuration
    # (e.g. capsules seated into the bracket) so they do not inject large corrective
    # impulses on the first step.
    _filter_initial_penetrating_pairs(builder)

    builder.color(balance_colors=False)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    return model, route_bodies


def _filter_initial_penetrating_pairs(builder: newton.ModelBuilder, penetration_eps: float = 1.0e-6) -> int:
    """Collide the initial pose and filter shape pairs that start in penetration.

    Detects contacts whose normal constraint ``C_n = thickness - dot(normal, p1 - p0)``
    is positive (overlapping) in the rest configuration and adds them to the builder's
    collision filter so they are excluded for the whole simulation. Returns the number
    of filtered pairs.
    """
    probe = builder.finalize()
    state = probe.state()
    contacts = probe.contacts()
    probe.collide(state, contacts)

    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0

    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    normal = contacts.rigid_contact_normal.numpy()[:n]
    m0 = contacts.rigid_contact_margin0.numpy()[:n]
    m1 = contacts.rigid_contact_margin1.numpy()[:n]
    shape_body = probe.shape_body.numpy()
    body_q = state.body_q.numpy()

    def xform_point(t, v):
        pos = t[:3]
        qx, qy, qz, qw = t[3:7]
        u = np.array([qx, qy, qz])
        uv = np.cross(u, v)
        return pos + v + 2.0 * (qw * uv + np.cross(u, uv))

    filtered: set[tuple[int, int]] = set()
    for i in range(n):
        s0 = int(shape0[i])
        s1 = int(shape1[i])
        if s0 < 0 or s1 < 0:
            continue
        b0 = int(shape_body[s0])
        b1 = int(shape_body[s1])
        cp0 = xform_point(body_q[b0], p0[i]) if b0 >= 0 else p0[i]
        cp1 = xform_point(body_q[b1], p1[i]) if b1 >= 0 else p1[i]
        c_n = (m0[i] + m1[i]) - float(np.dot(normal[i], cp1 - cp0))
        if c_n > penetration_eps:
            filtered.add((min(s0, s1), max(s0, s1)))

    for sa, sb in filtered:
        builder.add_shape_collision_filter_pair(sa, sb)
    return len(filtered)


def _route_points(
    cable: CableInstance,
    geometry: TrayGeometry,
    port_order: list[str],
    lane_y: float,
    stack_z: float,
) -> list[tuple[float, float, float]]:
    source_endpoint = endpoint_position(cable.connection.source, geometry, port_order, cable.cable_index)
    target_endpoint = endpoint_position(cable.connection.target, geometry, port_order, cable.cable_index)
    # The final connector fanout needs measured tray CAD.  Until then, seed
    # each cable directly into its assigned lane so VBD starts from a
    # collision-free bundle instead of resolving artificial connector braids.
    layer_offset_z = max(0.0, stack_z - (geometry.layer_floor_z + geometry.cable_radius))
    source = (source_endpoint[0], lane_y, source_endpoint[2] + layer_offset_z)
    target = (target_endpoint[0], lane_y, target_endpoint[2] + layer_offset_z)

    points: list[tuple[float, float, float]] = []
    points.extend(_smooth_yz(source[0], source[1], source[2], lane_y, stack_z, samples=7))
    points.append((target[0], lane_y, stack_z))
    points.extend(_smooth_yz(target[0], lane_y, stack_z, target[1], target[2], samples=7)[1:])
    return points


def _bracket_array_route_points(
    row: int,
    column: int,
    geometry: BracketArrayGeometry,
) -> list[tuple[float, float, float]]:
    column_offset = (float(column) - 0.5 * float(geometry.columns - 1)) * geometry.cable_pitch
    row_offset = (float(row) - 0.5 * float(geometry.rows - 1)) * geometry.cable_pitch
    z0 = geometry.bracket_top_z + geometry.cable_radius
    peak_height = geometry.vertical_lead + geometry.arch_height
    path_length_estimate = 2.0 * peak_height + math.hypot(geometry.plug_separation_x, geometry.receiver_offset_y)
    centerline_samples = max(32, int(math.ceil(path_length_estimate / geometry.route_segment_length)) + 1)

    centerline: list[tuple[float, float, float]] = []
    for sample_index in range(centerline_samples):
        s = float(sample_index) / float(centerline_samples - 1)
        transverse = _smootherstep(s)
        centerline.append(
            (
                geometry.plug_separation_x * transverse,
                geometry.receiver_offset_y * transverse,
                z0 + peak_height * math.sin(math.pi * s),
            )
        )

    points: list[tuple[float, float, float]] = []
    for point_index, center in enumerate(centerline):
        a = centerline[max(point_index - 1, 0)]
        b = centerline[min(point_index + 1, len(centerline) - 1)]
        tangent = _normalize((b[0] - a[0], b[1] - a[1], b[2] - a[2]))
        normal = _normalize((0.0, -tangent[2], tangent[1]))
        if _norm(normal) <= 1.0e-8:
            normal = (0.0, 1.0, 0.0)
        points.append(
            (
                center[0] + column_offset,
                center[1] + row_offset * normal[1],
                center[2] + row_offset * normal[2],
            )
        )
    return points


def _vertical_points(
    x: float,
    y: float,
    z0: float,
    z1: float,
    *,
    samples: int,
) -> list[tuple[float, float, float]]:
    return [(x, y, z0 + (z1 - z0) * float(i) / float(samples - 1)) for i in range(samples)]


def _smootherstep(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _cubic_bezier(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    *,
    samples: int,
) -> list[tuple[float, float, float]]:
    points = []
    for i in range(samples):
        t = float(i) / float(samples - 1)
        u = 1.0 - t
        points.append(
            (
                u**3 * p0[0] + 3.0 * u * u * t * p1[0] + 3.0 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3.0 * u * u * t * p1[1] + 3.0 * u * t * t * p2[1] + t**3 * p3[1],
                u**3 * p0[2] + 3.0 * u * u * t * p1[2] + 3.0 * u * t * t * p2[2] + t**3 * p3[2],
            )
        )
    return points


def _add_bracket_array_visuals(builder: newton.ModelBuilder, geometry: BracketArrayGeometry) -> None:
    block_color = (0.025, 0.023, 0.020)
    rail_color = (0.55, 0.56, 0.55)
    hx = 0.5 * geometry.array_width + geometry.bracket_margin
    hy = 0.5 * geometry.array_depth + geometry.bracket_margin
    hz = 0.5 * geometry.bracket_height
    z = geometry.bracket_top_z - hz
    source_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
        is_kinematic=True,
        label="source_bracket_body",
    )
    builder.add_shape_box(
        body=source_body,
        hx=hx,
        hy=hy,
        hz=hz,
        color=block_color,
        label="source_bracket",
    )

    receiver_body = builder.add_body(
        xform=wp.transform(wp.vec3(geometry.plug_separation_x, geometry.receiver_offset_y, z), wp.quat_identity()),
        is_kinematic=True,
        label="receiver_bracket_body",
    )
    builder.add_shape_box(
        body=receiver_body,
        hx=hx,
        hy=hy,
        hz=hz,
        color=block_color,
        label="receiver_bracket",
    )

    rail_center_x = 0.5 * geometry.plug_separation_x
    rail_center_y = 0.5 * geometry.receiver_offset_y
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(rail_center_x, rail_center_y, 0.001), wp.quat_identity()),
        hx=0.5 * geometry.plug_separation_x + hx,
        hy=hy + 0.006,
        hz=0.001,
        color=rail_color,
        label="tray_rail",
    )


def _smooth_yz(
    x: float,
    y0: float,
    z0: float,
    y1: float,
    z1: float,
    *,
    samples: int,
) -> list[tuple[float, float, float]]:
    pts = []
    for i in range(samples):
        t = float(i) / float(samples - 1)
        s = t * t * (3.0 - 2.0 * t)
        pts.append((x, y0 + (y1 - y0) * s, z0 + (z1 - z0) * s))
    return pts


def _deduplicate_points(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    deduped: list[tuple[float, float, float]] = []
    for point in points:
        if not deduped or _distance(deduped[-1], point) > 1.0e-8:
            deduped.append(point)
    return deduped


def resample_polyline(
    points: list[tuple[float, float, float]],
    max_segment_length: float,
) -> list[tuple[float, float, float]]:
    resampled: list[tuple[float, float, float]] = []
    for a, b in pairwise(points):
        if not resampled:
            resampled.append(a)
        length = _distance(a, b)
        steps = max(1, int(math.ceil(length / max_segment_length)))
        for i in range(1, steps + 1):
            t = float(i) / float(steps)
            resampled.append(
                (
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                )
            )
    return resampled


def polyline_length(points: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]]) -> float:
    return sum(_distance(a, b) for a, b in pairwise(points))


def compute_bend_radii(
    points: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]],
) -> list[float | None]:
    if len(points) < 3:
        return [None for _ in points]

    radii: list[float | None] = [None]
    for a, b, c in zip(points[:-2], points[1:-1], points[2:], strict=True):
        ab = _distance(a, b)
        bc = _distance(b, c)
        ca = _distance(c, a)
        cross_norm = _cross_norm(_sub(b, a), _sub(c, a))
        area_twice = cross_norm
        if area_twice <= 1.0e-12:
            radii.append(math.inf)
        else:
            radii.append((ab * bc * ca) / (2.0 * area_twice))
    radii.append(None)
    return radii


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def segment_distance(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float],
    p2: tuple[float, float, float],
    q2: tuple[float, float, float],
) -> float:
    """Return shortest distance between two 3D segments."""

    d1 = _sub(q1, p1)
    d2 = _sub(q2, p2)
    r = _sub(p1, p2)
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, r)
    eps = 1.0e-12

    if a <= eps and e <= eps:
        return _distance(p1, p2)
    if a <= eps:
        s = 0.0
        t = _clamp(f / e, 0.0, 1.0)
    else:
        c = _dot(d1, r)
        if e <= eps:
            t = 0.0
            s = _clamp(-c / a, 0.0, 1.0)
        else:
            b = _dot(d1, d2)
            denom = a * e - b * b
            s = _clamp((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            tnom = b * s + f
            if tnom < 0.0:
                t = 0.0
                s = _clamp(-c / a, 0.0, 1.0)
            elif tnom > e:
                t = 1.0
                s = _clamp((b - c) / a, 0.0, 1.0)
            else:
                t = tnom / e

    c1 = (p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s)
    c2 = (p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t)
    return _distance(c1, c2)


def _format_segment_pair(pair: tuple[int, int, int, int] | None) -> str | None:
    if pair is None:
        return None
    route_i, seg_i, route_j, seg_j = pair
    return f"route {route_i} seg {seg_i} vs route {route_j} seg {seg_j}"


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _norm(a)
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / norm, a[1] / norm, a[2] / norm)


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross_norm(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return math.sqrt(x * x + y * y + z * z)


ORDER_STRATEGIES = ("input", "shortest-first", "longest-first", "grouped-by-port", "grouped-by-wafer", "random")
LAYOUTS = ("bracket-array", "netlist")
DRIVE_AXES = ("x", "y", "z")


@wp.kernel
def _drive_kinematic_bodies(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_ids: wp.array[int],
    rest_q: wp.array[wp.transform],
    offset: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
):
    tid = wp.tid()
    body_id = body_ids[tid]
    tf = rest_q[tid]
    pos = wp.transform_get_translation(tf) + offset[0]
    rot = wp.transform_get_rotation(tf)
    body_q[body_id] = wp.transform(pos, rot)
    body_qd[body_id] = wp.spatial_vector(velocity[0], wp.vec3(0.0))


@wp.kernel
def _twist_kinematic_bodies(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_ids: wp.array[int],
    rest_q: wp.array[wp.transform],
    pivot: wp.array[wp.vec3],
    angle: wp.array[float],
    angular_speed: wp.array[float],
):
    tid = wp.tid()
    body_id = body_ids[tid]
    axis = wp.vec3(0.0, 0.0, 1.0)
    drive_rot = wp.quat_from_axis_angle(axis, angle[0])
    rest_tf = rest_q[tid]
    rest_pos = wp.transform_get_translation(rest_tf)
    rest_rot = wp.transform_get_rotation(rest_tf)
    arm = wp.quat_rotate(drive_rot, rest_pos - pivot[0])
    omega = axis * angular_speed[0]

    body_q[body_id] = wp.transform(pivot[0] + arm, wp.normalize(wp.mul(drive_rot, rest_rot)))
    body_qd[body_id] = wp.spatial_vector(wp.cross(omega, arm), omega)


@wp.kernel
def _twist_kinematic_bodies_from_time(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_ids: wp.array[int],
    rest_q: wp.array[wp.transform],
    pivot: wp.array[wp.vec3],
    time: wp.array[float],
    substep_time_offset: float,
    amplitude: float,
    frequency: float,
):
    tid = wp.tid()
    body_id = body_ids[tid]
    axis = wp.vec3(0.0, 0.0, 1.0)
    drive_time = time[0] + substep_time_offset
    omega_scale = 2.0 * math.pi * frequency
    phase = omega_scale * drive_time
    angle = amplitude * wp.sin(phase)
    angular_speed = omega_scale * amplitude * wp.cos(phase)
    drive_rot = wp.quat_from_axis_angle(axis, angle)
    rest_tf = rest_q[tid]
    rest_pos = wp.transform_get_translation(rest_tf)
    rest_rot = wp.transform_get_rotation(rest_tf)
    arm = wp.quat_rotate(drive_rot, rest_pos - pivot[0])
    omega = axis * angular_speed

    body_q[body_id] = wp.transform(pivot[0] + arm, wp.normalize(wp.mul(drive_rot, rest_rot)))
    body_qd[body_id] = wp.spatial_vector(wp.cross(omega, arm), omega)


@wp.kernel
def _advance_time(time: wp.array[float], dt: float):
    time[0] = time[0] + dt


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(args, "substeps", 4))
        self.sim_iterations = int(getattr(args, "iterations", 32))
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.layout = getattr(args, "layout", "bracket-array")
        self.bracket_geometry = None
        if self.layout == "bracket-array":
            self.geometry = BracketArrayGeometry(
                rows=int(getattr(args, "bracket_rows", 8)),
                columns=int(getattr(args, "bracket_columns", 12)),
                cable_pitch=float(getattr(args, "cable_pitch_mm", 2.0)) * 1.0e-3,
                vertical_lead=float(getattr(args, "vertical_lead_mm", 50.0)) * 1.0e-3,
                arch_height=float(getattr(args, "arch_height_mm", 55.0)) * 1.0e-3,
                plug_separation_x=float(getattr(args, "plug_separation_x_mm", 0.0)) * 1.0e-3,
                receiver_offset_y=float(getattr(args, "receiver_offset_y_mm", 145.0)) * 1.0e-3,
                route_segment_length=float(getattr(args, "route_segment_length_mm", 5.0)) * 1.0e-3,
            )
            self.bracket_geometry = self.geometry
            self.netlist = None
            self.routes, self.route_metrics = generate_bracket_array_routes(
                self.geometry,
                cable_limit=int(getattr(args, "cable_limit", self.geometry.cable_count)),
            )
        else:
            self.geometry = TrayGeometry(
                rack_width=float(getattr(args, "rack_width_mm", 600.0)) * 1.0e-3,
                rack_depth=float(getattr(args, "rack_depth_mm", 1068.0)) * 1.0e-3,
                rack_height=float(getattr(args, "rack_height_mm", 2495.0)) * 1.0e-3,
            )

            netlist_csv = getattr(args, "netlist_csv", None)
            if netlist_csv:
                self.netlist = load_netlist_csv(netlist_csv)
            else:
                self.netlist = load_netlist_csv(io.StringIO(create_demo_netlist_csv()))

            self.routes, self.route_metrics = generate_routes(
                self.netlist,
                self.geometry,
                strategy=getattr(args, "order_strategy", "input"),
                seed=int(getattr(args, "seed", 0)),
                cable_limit=int(getattr(args, "cable_limit", 8)),
            )
        self.clearance_metrics = compute_route_clearance(self.routes, cable_radius=self.geometry.cable_radius)

        self.model, self.route_bodies = build_vbd_model(
            self.routes,
            cable_radius=self.geometry.cable_radius,
            bracket_geometry=self.bracket_geometry,
            bend_stiffness=float(getattr(args, "bend_stiffness", 5.0e5)),
            bend_damping=float(getattr(args, "bend_damping", 1.0e-4)),
            stretch_stiffness=float(getattr(args, "stretch_stiffness", 1.0e10)),
            stretch_damping=float(getattr(args, "stretch_damping", 0.0)),
        )
        contact_budget_per_shape = int(getattr(args, "contact_budget_per_shape", 128))
        self.model.rigid_contact_max = max(1000, self.model.shape_count * contact_budget_per_shape)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=float(getattr(args, "friction_epsilon", 1.0e-4)),
            rigid_body_contact_buffer_size=int(getattr(args, "rigid_body_contact_buffer_size", 256)),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.graph = None

        self.drive_source_twist = bool(getattr(args, "drive_source_twist", True))
        self.source_twist_amplitude = math.radians(float(getattr(args, "source_twist_amplitude_deg", 40.0)))
        self.source_twist_frequency = float(getattr(args, "source_twist_frequency_hz", 0.7))
        self.drive_receiver = bool(getattr(args, "drive_receiver", False))
        self.receiver_drive_axis = getattr(args, "receiver_drive_axis", "y")
        if self.receiver_drive_axis not in DRIVE_AXES:
            raise ValueError(f"Unsupported receiver drive axis: {self.receiver_drive_axis}")
        self.receiver_drive_amplitude = float(getattr(args, "receiver_drive_amplitude_mm", 0.0)) * 1.0e-3
        self.receiver_drive_frequency = float(getattr(args, "receiver_drive_frequency_hz", 0.8))
        self.source_anchor_bodies = [bodies[0] for bodies in self.route_bodies if bodies]
        self.receiver_anchor_bodies = [bodies[-1] for bodies in self.route_bodies if bodies]
        self.source_bracket_body = self._find_body_label("source_bracket_body")
        self.receiver_bracket_body = self._find_body_label("receiver_bracket_body")
        self._source_driven_body_ids = self.source_anchor_bodies.copy()
        if self.source_bracket_body is not None:
            self._source_driven_body_ids.append(self.source_bracket_body)
        self._receiver_driven_body_ids = self.receiver_anchor_bodies.copy()
        if self.receiver_bracket_body is not None:
            self._receiver_driven_body_ids.append(self.receiver_bracket_body)
        self.static_anchor_bodies = []
        if not self.drive_source_twist:
            self.static_anchor_bodies.extend(self.source_anchor_bodies)
        if not self.drive_receiver:
            self.static_anchor_bodies.extend(self.receiver_anchor_bodies)

        body_q0 = self.state_0.body_q.numpy()
        self._static_anchor_q0 = body_q0[self.static_anchor_bodies].copy() if self.static_anchor_bodies else None
        self._source_driven_body_q0 = (
            body_q0[self._source_driven_body_ids].copy() if self._source_driven_body_ids else None
        )
        self._receiver_driven_body_q0 = (
            body_q0[self._receiver_driven_body_ids].copy() if self._receiver_driven_body_ids else None
        )
        self._source_driven_body_ids_wp = (
            wp.array(self._source_driven_body_ids, dtype=int, device=self.model.device)
            if self._source_driven_body_ids
            else None
        )
        self._receiver_driven_body_ids_wp = (
            wp.array(self._receiver_driven_body_ids, dtype=int, device=self.model.device)
            if self._receiver_driven_body_ids
            else None
        )
        self._source_driven_body_rest_q_wp = None
        if self._source_driven_body_q0 is not None:
            self._source_driven_body_rest_q_wp = wp.array(
                [_np_transform_to_wp(body_q) for body_q in self._source_driven_body_q0],
                dtype=wp.transform,
                device=self.model.device,
            )
        self._receiver_driven_body_rest_q_wp = None
        if self._receiver_driven_body_q0 is not None:
            self._receiver_driven_body_rest_q_wp = wp.array(
                [_np_transform_to_wp(body_q) for body_q in self._receiver_driven_body_q0],
                dtype=wp.transform,
                device=self.model.device,
            )
        z0 = self.geometry.bracket_top_z + self.geometry.cable_radius if self.layout == "bracket-array" else 0.0
        self._source_twist_pivot_np = np.array([0.0, 0.0, z0], dtype=np.float32)
        self._source_twist_pivot = wp.array(
            [wp.vec3(*self._source_twist_pivot_np)], dtype=wp.vec3, device=self.model.device
        )
        self._source_twist_angle = wp.zeros(1, dtype=float, device=self.model.device)
        self._source_twist_angular_speed = wp.zeros(1, dtype=float, device=self.model.device)
        self._graph_drive_time = wp.zeros(1, dtype=float, device=self.model.device)
        self._last_source_twist_angle = 0.0
        self._receiver_drive_offset = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self._receiver_drive_velocity = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self._last_receiver_drive_offset_np = np.zeros(3, dtype=np.float32)
        self._initial_body_positions = self.state_0.body_q.numpy()[:, :3].copy()
        self.stability_metrics = {
            "max_abs_position_m": 0.0,
            "min_body_z_m": float("inf"),
            "max_state_rate": 0.0,
            "max_connector_drift_m": 0.0,
            "max_source_twist_angle_rad": 0.0,
            "max_source_anchor_displacement_m": 0.0,
            "max_source_anchor_error_m": 0.0,
            "max_receiver_drive_displacement_m": 0.0,
            "max_receiver_anchor_error_m": 0.0,
            "max_downward_displacement_m": 0.0,
            "max_rigid_contact_count": 0,
        }
        self._update_metrics()

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(-0.26, 0.07, 0.12), pitch=-15.0, yaw=-10.0)

        self.capture()

    def _find_body_label(self, label: str) -> int | None:
        for body_index, body_label in enumerate(getattr(self.model, "body_label", ())):
            if body_label == label:
                return body_index
        return None

    def capture(self):
        """Capture simulation loop into a CUDA graph for interactive runs."""
        if self.viewer is None or self.drive_receiver or not self.solver.device.is_cuda:
            self.graph = None
            return

        self._graph_drive_time.zero_()
        with wp.ScopedCapture() as capture:
            self.simulate(graph_time=self._graph_drive_time if self.drive_source_twist else None)
        self.graph = capture.graph

    def simulate(self, graph_time: wp.array | None = None):
        for substep in range(self.sim_substeps):
            drive_time = self.sim_time + float(substep) * self.sim_dt
            if graph_time is None:
                self._apply_source_twist(drive_time)
            else:
                self._apply_source_twist_from_graph_time(graph_time, float(substep) * self.sim_dt)
            self._apply_receiver_drive(drive_time)
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        if graph_time is not None:
            wp.launch(kernel=_advance_time, dim=1, inputs=(graph_time, self.frame_dt), device=self.model.device)

    def step(self):
        drive_time = self.sim_time + float(self.sim_substeps - 1) * self.sim_dt
        if self.graph:
            wp.capture_launch(self.graph)
            if self.drive_source_twist:
                self._last_source_twist_angle = self.source_twist_amplitude * math.sin(
                    2.0 * math.pi * self.source_twist_frequency * drive_time
                )
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._update_metrics()

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _apply_source_twist(self, time: float):
        if (
            not self.drive_source_twist
            or self._source_driven_body_ids_wp is None
            or self._source_driven_body_rest_q_wp is None
            or self.state_0.body_q is None
            or self.state_0.body_qd is None
        ):
            return

        phase = 2.0 * math.pi * self.source_twist_frequency * time
        angle = self.source_twist_amplitude * math.sin(phase)
        angular_speed = 2.0 * math.pi * self.source_twist_frequency * self.source_twist_amplitude * math.cos(phase)
        self._last_source_twist_angle = angle
        self._source_twist_angle.assign([angle])
        self._source_twist_angular_speed.assign([angular_speed])
        wp.launch(
            kernel=_twist_kinematic_bodies,
            dim=len(self._source_driven_body_ids),
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self._source_driven_body_ids_wp,
                self._source_driven_body_rest_q_wp,
                self._source_twist_pivot,
                self._source_twist_angle,
                self._source_twist_angular_speed,
            ),
            device=self.model.device,
        )

    def _apply_source_twist_from_graph_time(self, graph_time: wp.array, substep_time_offset: float):
        if (
            not self.drive_source_twist
            or self._source_driven_body_ids_wp is None
            or self._source_driven_body_rest_q_wp is None
            or self.state_0.body_q is None
            or self.state_0.body_qd is None
        ):
            return

        wp.launch(
            kernel=_twist_kinematic_bodies_from_time,
            dim=len(self._source_driven_body_ids),
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self._source_driven_body_ids_wp,
                self._source_driven_body_rest_q_wp,
                self._source_twist_pivot,
                graph_time,
                substep_time_offset,
                self.source_twist_amplitude,
                self.source_twist_frequency,
            ),
            device=self.model.device,
        )

    def _apply_receiver_drive(self, time: float):
        if (
            not self.drive_receiver
            or self._receiver_driven_body_ids_wp is None
            or self._receiver_driven_body_rest_q_wp is None
            or self.state_0.body_q is None
            or self.state_0.body_qd is None
        ):
            return

        phase = 2.0 * math.pi * self.receiver_drive_frequency * time
        displacement = self.receiver_drive_amplitude * math.sin(phase)
        speed = 2.0 * math.pi * self.receiver_drive_frequency * self.receiver_drive_amplitude * math.cos(phase)
        offset = np.zeros(3, dtype=np.float32)
        velocity = np.zeros(3, dtype=np.float32)
        axis_index = DRIVE_AXES.index(self.receiver_drive_axis)
        offset[axis_index] = displacement
        velocity[axis_index] = speed
        self._last_receiver_drive_offset_np = offset.copy()
        self._receiver_drive_offset.assign([wp.vec3(*offset)])
        self._receiver_drive_velocity.assign([wp.vec3(*velocity)])
        wp.launch(
            kernel=_drive_kinematic_bodies,
            dim=len(self._receiver_driven_body_ids),
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self._receiver_driven_body_ids_wp,
                self._receiver_driven_body_rest_q_wp,
                self._receiver_drive_offset,
                self._receiver_drive_velocity,
            ),
            device=self.model.device,
        )

    def test_final(self):
        """Verify that the generated NV72 cable model stayed numerically stable."""
        self._update_metrics()
        self._assert_stable_state()
        assert len(self.routes) == int(self.route_metrics["cable_count"]), "Route metric cable count mismatch"
        assert self.model.body_count > len(self.routes), "Expected rod segments, not one body per cable"

    def test_post_step(self):
        """Catch non-physical behavior as soon as it appears during example tests."""
        self._update_metrics()
        self._assert_stable_state()

    def _update_metrics(self):
        if self.state_0.body_q is None or self.state_0.body_qd is None:
            return

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        positions = body_q[:, :3]

        if body_q.size:
            self.stability_metrics["max_abs_position_m"] = max(
                float(self.stability_metrics["max_abs_position_m"]),
                float(np.max(np.abs(positions))),
            )
            self.stability_metrics["min_body_z_m"] = min(
                float(self.stability_metrics["min_body_z_m"]),
                float(np.min(positions[:, 2])),
            )
        if body_qd.size:
            state_rates = np.linalg.norm(body_qd, axis=1)
            self.stability_metrics["max_state_rate"] = max(
                float(self.stability_metrics["max_state_rate"]),
                float(np.max(state_rates)),
            )
        if body_q.shape[0] == self._initial_body_positions.shape[0]:
            downward = self._initial_body_positions[:, 2] - positions[:, 2]
            self.stability_metrics["max_downward_displacement_m"] = max(
                float(self.stability_metrics["max_downward_displacement_m"]),
                float(np.max(downward)),
            )

        if self.static_anchor_bodies and self._static_anchor_q0 is not None:
            pinned_now = body_q[self.static_anchor_bodies]
            drift = np.linalg.norm(pinned_now[:, :3] - self._static_anchor_q0[:, :3], axis=1)
            self.stability_metrics["max_connector_drift_m"] = max(
                float(self.stability_metrics["max_connector_drift_m"]),
                float(np.max(drift)),
            )
        if self.drive_source_twist and self._source_driven_body_ids and self._source_driven_body_q0 is not None:
            source_now = body_q[self._source_driven_body_ids]
            expected = self._rotate_source_points(self._source_driven_body_q0[:, :3], self._last_source_twist_angle)
            source_offsets = source_now[:, :3] - self._source_driven_body_q0[:, :3]
            source_error = np.linalg.norm(source_now[:, :3] - expected, axis=1)
            self.stability_metrics["max_source_twist_angle_rad"] = max(
                float(self.stability_metrics["max_source_twist_angle_rad"]),
                abs(float(self._last_source_twist_angle)),
            )
            self.stability_metrics["max_source_anchor_displacement_m"] = max(
                float(self.stability_metrics["max_source_anchor_displacement_m"]),
                float(np.max(np.linalg.norm(source_offsets, axis=1))),
            )
            self.stability_metrics["max_source_anchor_error_m"] = max(
                float(self.stability_metrics["max_source_anchor_error_m"]),
                float(np.max(source_error)),
            )
        if self.drive_receiver and self._receiver_driven_body_ids and self._receiver_driven_body_q0 is not None:
            driven_now = body_q[self._receiver_driven_body_ids]
            driven_offsets = driven_now[:, :3] - self._receiver_driven_body_q0[:, :3]
            receiver_error = np.linalg.norm(driven_offsets - self._last_receiver_drive_offset_np, axis=1)
            self.stability_metrics["max_receiver_drive_displacement_m"] = max(
                float(self.stability_metrics["max_receiver_drive_displacement_m"]),
                float(np.max(np.linalg.norm(driven_offsets, axis=1))),
            )
            self.stability_metrics["max_receiver_anchor_error_m"] = max(
                float(self.stability_metrics["max_receiver_anchor_error_m"]),
                float(np.max(receiver_error)),
            )

        contact_count = getattr(self.contacts, "rigid_contact_count", None)
        if contact_count is not None:
            self.stability_metrics["max_rigid_contact_count"] = max(
                int(self.stability_metrics["max_rigid_contact_count"]),
                int(contact_count.numpy()[0]),
            )

    def _rotate_source_points(self, points: np.ndarray, angle: float) -> np.ndarray:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        shifted = points - self._source_twist_pivot_np
        rotated = shifted.copy()
        rotated[:, 0] = shifted[:, 0] * cos_a - shifted[:, 1] * sin_a
        rotated[:, 1] = shifted[:, 0] * sin_a + shifted[:, 1] * cos_a
        return rotated + self._source_twist_pivot_np

    def _assert_stable_state(self):
        if self.state_0.body_q is None or self.state_0.body_qd is None:
            raise RuntimeError("Body state is not available.")

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.isfinite(body_q).all(), "NV72 tray produced non-finite body poses"
        assert np.isfinite(body_qd).all(), "NV72 tray produced non-finite body velocities"
        assert self.clearance_metrics["min_clearance_m"] >= 0.0, (
            "Seed route starts in cable collision: "
            f"{self.clearance_metrics['min_clearance_m']:.6e} m at {self.clearance_metrics['closest_pair']}"
        )
        assert self.clearance_metrics["min_self_clearance_m"] >= 0.0, (
            "Seed route self-collides: "
            f"{self.clearance_metrics['min_self_clearance_m']:.6e} m at "
            f"{self.clearance_metrics['closest_self_pair']}"
        )
        assert self.stability_metrics["max_abs_position_m"] < 10.0, (
            f"NV72 tray body position exceeded tray-scale bounds: {self.stability_metrics['max_abs_position_m']:.6e} m"
        )
        assert self.stability_metrics["min_body_z_m"] > -0.005, (
            f"NV72 tray cable fell below the tray floor: {self.stability_metrics['min_body_z_m']:.6e} m"
        )
        assert self.stability_metrics["max_state_rate"] < 500.0, (
            f"NV72 tray state rate looks explosive: {self.stability_metrics['max_state_rate']:.6e}"
        )
        assert self.stability_metrics["max_downward_displacement_m"] < 0.050, (
            "NV72 bracket bundle is not self-supporting enough: "
            f"{self.stability_metrics['max_downward_displacement_m']:.6e} m downward displacement"
        )
        assert self.stability_metrics["max_connector_drift_m"] < 1.0e-4, (
            f"Static connector lead-in bodies drifted by {self.stability_metrics['max_connector_drift_m']:.6e} m"
        )
        assert self.stability_metrics["max_source_anchor_error_m"] < 1.0e-4, (
            "Driven source twist bodies missed their prescribed motion by "
            f"{self.stability_metrics['max_source_anchor_error_m']:.6e} m"
        )
        assert self.stability_metrics["max_receiver_anchor_error_m"] < 1.0e-4, (
            "Driven receiver bodies missed their prescribed motion by "
            f"{self.stability_metrics['max_receiver_anchor_error_m']:.6e} m"
        )
        assert self.stability_metrics["max_rigid_contact_count"] <= self.model.rigid_contact_max, (
            "NV72 tray exhausted the rigid contact buffer: "
            f"{self.stability_metrics['max_rigid_contact_count']} > {self.model.rigid_contact_max}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--layout", default="bracket-array", choices=LAYOUTS, help="Cable layout to instantiate")
        parser.add_argument("--netlist-csv", default=None, help="Path to production netlist-mapping.csv")
        parser.add_argument("--cable-limit", type=int, default=96, help="Number of physical cables to instantiate")
        parser.add_argument("--order-strategy", default="input", choices=ORDER_STRATEGIES, help="Placement order")
        parser.add_argument("--seed", type=int, default=0, help="Random seed for random placement order")
        parser.add_argument("--bracket-rows", type=int, default=8, help="Connector cable rows")
        parser.add_argument("--bracket-columns", type=int, default=12, help="Connector cable columns")
        parser.add_argument("--cable-pitch-mm", type=float, default=2.0, help="Center spacing in the bracket array")
        parser.add_argument("--vertical-lead-mm", type=float, default=50.0, help="Vertical self-supported lead-out")
        parser.add_argument("--arch-height-mm", type=float, default=55.0, help="Height added above the vertical lead")
        parser.add_argument("--plug-separation-x-mm", type=float, default=0.0, help="Receiver plug X offset")
        parser.add_argument("--receiver-offset-y-mm", type=float, default=145.0, help="Source-to-receiver plug offset")
        parser.add_argument("--route-segment-length-mm", type=float, default=5.0, help="Maximum rod segment length")
        parser.add_argument(
            "--drive-receiver",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Animate the receiver-side connector anchors",
        )
        parser.add_argument(
            "--drive-source-twist",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Twist the source connector anchors around the bracket normal",
        )
        parser.add_argument("--source-twist-amplitude-deg", type=float, default=40.0, help="Source twist amplitude")
        parser.add_argument("--source-twist-frequency-hz", type=float, default=0.7, help="Source twist frequency")
        parser.add_argument(
            "--receiver-drive-axis",
            default="y",
            choices=DRIVE_AXES,
            help="Axis used for the receiver connector sinusoidal drive",
        )
        parser.add_argument("--receiver-drive-amplitude-mm", type=float, default=0.0, help="Receiver drive amplitude")
        parser.add_argument("--receiver-drive-frequency-hz", type=float, default=0.8, help="Receiver drive frequency")
        parser.add_argument("--rack-width-mm", type=float, default=600.0, help="Inferred tray/rack width")
        parser.add_argument("--rack-depth-mm", type=float, default=1068.0, help="Inferred tray/rack depth")
        parser.add_argument("--rack-height-mm", type=float, default=2495.0, help="Inferred tray/rack height")
        parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps per frame")
        parser.add_argument("--iterations", type=int, default=32, help="SolverVBD iterations")
        parser.add_argument("--bend-stiffness", type=float, default=5.0e5, help="Rod bend stiffness")
        parser.add_argument("--bend-damping", type=float, default=1.0e-4, help="Rod bend damping")
        parser.add_argument("--stretch-stiffness", type=float, default=1.0e10, help="Rod stretch stiffness")
        parser.add_argument("--stretch-damping", type=float, default=0.0, help="Rod stretch damping")
        parser.add_argument("--friction-epsilon", type=float, default=1.0e-4, help="Solver friction epsilon")
        parser.add_argument("--rigid-body-contact-buffer-size", type=int, default=512)
        parser.add_argument(
            "--contact-budget-per-shape",
            type=int,
            default=128,
            help="Rigid contact capacity multiplier for dense cable bundles",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
