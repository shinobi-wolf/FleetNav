"""Pygame visualization for the EVRP simulator.

Shows the road network with speed-based edge coloring (red=slow, green=fast),
directed arrows on edges, node types, and current vehicle positions.
Updates every time a vehicle arrives at a stop (not mid-leg).
"""
from __future__ import annotations

import math
from typing import Any

try:
    import pygame
except ImportError as e:
    raise ImportError("pygame is required for visualization") from e

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SCREEN_W = 1200
SCREEN_H = 800
MARGIN = 50
PANEL_W = 320

MAP_W = SCREEN_W - PANEL_W  # left side for the network

# Colors
BG          = (245, 245, 245)
EDGE_COL    = (180, 180, 180)
NODE_COL    = (90, 90, 90)
DEPOT_COL   = (30, 100, 210)
CUSTOMER_COL        = (210, 60, 40)
CUSTOMER_SERVED_COL = (50, 180, 80)
AFS_COL     = (140, 60, 200)
TEXT_DARK   = (20, 20, 20)
TEXT_MID    = (80, 80, 80)
PANEL_BG    = (230, 230, 235)

VEHICLE_COLORS = [
    (255, 140, 0),
    (0, 180, 220),
    (220, 0, 180),
    (0, 200, 80),
    (255, 60, 60),
    (60, 60, 255),
    (200, 200, 0),
    (0, 200, 200),
]

_EDGE_OFFSET_PX = 3   # perpendicular offset for directed edges (px)
_ARROW_SIZE     = 8   # arrowhead length (px)
_ARROW_HALF_W   = 3   # arrowhead half-width at base (px)


class PygameRenderer:
    """Lightweight pygame window for EVRP simulation.

    Parameters
    ----------
    net : sumolib net object
    node_roles : dict[str, str]
        Maps sumo_node_id → 'depot' | 'customer' | 'afs'
    node_labels : dict[str, str] | None
        Maps sumo_node_id → instance node label (e.g., D, C1, F3).
    vehicle_ids : list[int]
    v_min_ms : float | None
        Min PSL speed in m/s (for edge color normalization). None = no coloring.
    v_max_ms : float | None
        Max PSL speed in m/s.
    fps : int
    """

    def __init__(
        self,
        net,
        node_roles: dict[str, str],
        node_labels: dict[str, str] | None,
        vehicle_ids: list[int],
        v_min_ms: float | None = None,
        v_max_ms: float | None = None,
        fps: int = 30,
    ) -> None:
        self._net = net
        self._node_roles = node_roles
        self._node_labels = node_labels or {}
        self._vehicle_ids = vehicle_ids
        self._v_min_ms = v_min_ms
        self._v_max_ms = v_max_ms
        self._fps = fps

        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        self._node_pos: dict[str, tuple[float, float]] = {}
        self._edges: list[tuple[str, str]] = []
        self._edge_ids: list[str] = []
        self._build_graph_geometry()
        self._transform = self._make_transform()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_graph_geometry(self) -> None:
        for node in self._net.getNodes():
            nid = node.getID()
            coord = node.getCoord()
            self._node_pos[nid] = (float(coord[0]), float(coord[1]))
        for edge in self._net.getEdges():
            eid = edge.getID()
            if eid.startswith(":"):
                continue
            self._edges.append((edge.getFromNode().getID(), edge.getToNode().getID()))
            self._edge_ids.append(eid)

    def _make_transform(self):
        xs = [p[0] for p in self._node_pos.values()]
        ys = [p[1] for p in self._node_pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        map_draw_w = MAP_W - 2 * MARGIN
        map_draw_h = SCREEN_H - 2 * MARGIN
        sx = map_draw_w / max(1.0, max_x - min_x)
        sy = map_draw_h / max(1.0, max_y - min_y)
        scale = min(sx, sy)

        def transform(x: float, y: float) -> tuple[int, int]:
            return (int(MARGIN + (x - min_x) * scale),
                    int(MARGIN + (max_y - y) * scale))

        return transform

    def _init_window(self) -> None:
        pygame.init()
        pygame.display.init()
        self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("TA-EVRP Simulation")
        self._clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        sim_time_s: float,
        edge_speeds: dict[str, float] | None,
        vehicle_positions: dict[int, str | tuple[float, float]],
        served_customers: set[str],
        vehicle_states: dict[int, Any],
        vehicle_speeds: dict[int, float] | None = None,
    ) -> None:
        """Redraw the window.

        Parameters
        ----------
        edge_speeds : dict[edge_id, speed_ms] or None
            Per-edge current speed in m/s. None means traffic is disabled
            (all edges drawn in the default gray).
        vehicle_speeds : dict[vehicle_id, speed_ms] or None
            Real-time speed per vehicle from TraCI. None means not available.
        """
        if self._screen is None:
            self._init_window()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        surf = self._screen
        surf.fill(BG)

        self._draw_edges(surf, edge_speeds)
        self._draw_nodes(surf, served_customers)
        self._draw_vehicles(surf, vehicle_positions)
        self._draw_panel(surf, sim_time_s, edge_speeds, vehicle_positions, served_customers, vehicle_states, vehicle_speeds)

        pygame.display.flip()
        self._clock.tick(self._fps)

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _speed_to_color(self, speed_ms: float) -> tuple[int, int, int]:
        """Red (slow) → green (fast) over the configured PSL speed range."""
        if self._v_min_ms is None or self._v_max_ms is None:
            return EDGE_COL
        v_range = self._v_max_ms - self._v_min_ms
        if v_range < 1e-9:
            return EDGE_COL
        t = (speed_ms - self._v_min_ms) / v_range
        t = max(0.0, min(1.0, t))
        return (int(255 * (1.0 - t)), int(255 * t), 0)

    @staticmethod
    def _draw_directed_edge(
        surf: pygame.Surface,
        color: tuple,
        p1: tuple[int, int],
        p2: tuple[int, int],
        width: int = 2,
    ) -> None:
        """Draw a directed edge: offset line + arrowhead in direction p1→p2."""
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1.0:
            return

        ux, uy = dx / length, dy / length
        ox, oy = -uy * _EDGE_OFFSET_PX, ux * _EDGE_OFFSET_PX

        ax, ay = x1 + ox, y1 + oy
        bx, by = x2 + ox, y2 + oy

        pygame.draw.line(surf, color, (int(ax), int(ay)), (int(bx), int(by)), width)

        if length > 2.5 * _ARROW_SIZE:
            t = 0.6
            tip_x = ax + (bx - ax) * t
            tip_y = ay + (by - ay) * t
            base_x = tip_x - ux * _ARROW_SIZE
            base_y = tip_y - uy * _ARROW_SIZE
            left  = (base_x - uy * _ARROW_HALF_W, base_y + ux * _ARROW_HALF_W)
            right = (base_x + uy * _ARROW_HALF_W, base_y - ux * _ARROW_HALF_W)
            pygame.draw.polygon(surf, color, [
                (int(tip_x), int(tip_y)),
                (int(left[0]),  int(left[1])),
                (int(right[0]), int(right[1])),
            ])

    def _draw_edges(self, surf: pygame.Surface, edge_speeds: dict[str, float] | None) -> None:
        traffic_on = edge_speeds is not None
        width = 3 if traffic_on else 2

        for (fn, tn), eid in zip(self._edges, self._edge_ids):
            if fn not in self._node_pos or tn not in self._node_pos:
                continue
            if traffic_on and eid in edge_speeds:
                color = self._speed_to_color(edge_speeds[eid])
            else:
                color = EDGE_COL
            p1 = self._transform(*self._node_pos[fn])
            p2 = self._transform(*self._node_pos[tn])
            self._draw_directed_edge(surf, color, p1, p2, width=width)

    def _draw_nodes(self, surf: pygame.Surface, served_customers: set[str]) -> None:
        font = pygame.font.Font(None, 16)
        font_small = pygame.font.Font(None, 14)
        for nid, pos in self._node_pos.items():
            role = self._node_roles.get(nid, "road")
            px, py = self._transform(*pos)

            if role == "depot":
                color, r = DEPOT_COL, 10
            elif role == "customer":
                color = CUSTOMER_SERVED_COL if nid in served_customers else CUSTOMER_COL
                r = 8
            elif role == "afs":
                color, r = AFS_COL, 9
            else:
                color, r = NODE_COL, 5

            pygame.draw.circle(surf, color, (px, py), r)

            if role in ("depot", "customer", "afs"):
                label = self._node_labels.get(nid)
                if not label:
                    label = {"depot": "D", "customer": "C", "afs": "F"}.get(role, "")
                if label:
                    txt_font = font_small if len(label) > 1 else font
                    txt = txt_font.render(label, True, (255, 255, 255))
                    surf.blit(txt, txt.get_rect(center=(px, py)))

    def _draw_vehicles(
        self,
        surf: pygame.Surface,
        vehicle_positions: dict[int, str | tuple[float, float]],
    ) -> None:
        font = pygame.font.Font(None, 18)
        for idx, (vid, pos) in enumerate(vehicle_positions.items()):
            color = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
            if isinstance(pos, tuple):
                px, py = self._transform(pos[0], pos[1])
            else:
                if pos not in self._node_pos:
                    continue
                px, py = self._transform(*self._node_pos[pos])

            px += idx * 14
            pygame.draw.circle(surf, color, (px, py - 18), 9)
            txt = font.render(str(vid), True, (255, 255, 255))
            surf.blit(txt, txt.get_rect(center=(px, py - 18)))

    def _draw_panel(
        self,
        surf: pygame.Surface,
        sim_time_s: float,
        edge_speeds: dict[str, float] | None,
        vehicle_positions: dict[int, str | tuple[float, float]],
        served_customers: set[str],
        vehicle_states: dict[int, Any],
        vehicle_speeds: dict[int, float] | None = None,
    ) -> None:
        panel_rect = pygame.Rect(MAP_W, 0, PANEL_W, SCREEN_H)
        pygame.draw.rect(surf, PANEL_BG, panel_rect)
        pygame.draw.line(surf, (180, 180, 190), (MAP_W, 0), (MAP_W, SCREEN_H), 2)

        font_h = pygame.font.Font(None, 28)
        font   = pygame.font.Font(None, 22)

        x = MAP_W + 16
        y = 16

        def line(text: str, big: bool = False, color: tuple = TEXT_DARK) -> None:
            nonlocal y
            f = font_h if big else font
            surf.blit(f.render(text, True, color), (x, y))
            y += 32 if big else 22

        sim_h = sim_time_s / 3600.0
        line(f"t = {sim_time_s:.0f}s  ({sim_h:.2f}h)", big=True)
        if edge_speeds:
            avg_kmh = sum(edge_speeds.values()) / len(edge_speeds) * 3.6
            line(f"Traffic avg: {avg_kmh:.1f} km/h")
        else:
            line("Traffic: disabled")

        total_c = sum(1 for r in self._node_roles.values() if r == "customer")
        line(f"Customers: {len(served_customers)}/{total_c}")

        y += 10
        line("── Legend ──", color=TEXT_MID)
        self._legend_dot(surf, x, y, DEPOT_COL,            "Depot")               ; y += 22
        self._legend_dot(surf, x, y, CUSTOMER_COL,         "Customer (pending)")  ; y += 22
        self._legend_dot(surf, x, y, CUSTOMER_SERVED_COL,  "Customer (served)")   ; y += 22
        self._legend_dot(surf, x, y, AFS_COL,              "AFS (fuel station)")  ; y += 22

        if edge_speeds is not None and self._v_min_ms is not None:
            y += 6
            line("── Traffic ──", color=TEXT_MID)
            self._legend_dot(surf, x, y, (255, 0, 0), f"Slow ({self._v_min_ms * 3.6:.0f} km/h)") ; y += 22
            self._legend_dot(surf, x, y, (0, 255, 0), f"Fast ({self._v_max_ms * 3.6:.0f} km/h)") ; y += 22

        y += 10
        line("── Vehicles ──", color=TEXT_MID)
        for idx, vid in enumerate(self._vehicle_ids):
            vs = vehicle_states.get(vid)
            color_v = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
            pos = vehicle_positions.get(vid, "?")
            loc_label = "transit" if isinstance(pos, tuple) else str(pos)
            self._legend_dot(surf, x, y, color_v, f"V{vid}  @{loc_label}")
            y += 22
            if vs is not None:
                spd_ms = (vehicle_speeds or {}).get(vid, 0.0)
                spd_kmh = spd_ms * 3.6
                line(f"  speed={spd_kmh:.1f} km/h")
                line(f"  dist={vs.distance_m:.0f}m  fuel={vs.current_fuel:.2f}gal")
                line(f"  served={vs.customers_served}  recharges={vs.num_recharges}")

    @staticmethod
    def _legend_dot(surf: pygame.Surface, x: int, y: int, color: tuple, label: str) -> None:
        font = pygame.font.Font(None, 20)
        pygame.draw.circle(surf, color, (x + 7, y + 7), 7)
        surf.blit(font.render(label, True, TEXT_DARK), (x + 20, y))
