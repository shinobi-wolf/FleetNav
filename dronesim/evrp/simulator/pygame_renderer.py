"""Pygame visualization for the EVRP simulator.

Shows the road network, node types, and current vehicle positions.
Updates every time a vehicle arrives at a stop (not mid-leg).
"""
from __future__ import annotations

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
CUSTOMER_COL        = (210, 60, 40)    # unserved
CUSTOMER_SERVED_COL = (50, 180, 80)    # served
AFS_COL     = (140, 60, 200)
TEXT_DARK   = (20, 20, 20)
TEXT_MID    = (80, 80, 80)
PANEL_BG    = (230, 230, 235)

# One bright color per vehicle (cycles if > 8 vehicles)
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


class PygameRenderer:
    """Lightweight pygame window for EVRP simulation.

    Parameters
    ----------
    net : sumolib net object
        Used to extract node positions and edges for drawing.
    node_roles : dict[str, str]
        Maps sumo_node_id → role: 'depot' | 'customer' | 'afs'
    vehicle_ids : list[int]
        Ordered list of vehicle IDs (for legend colours).
    fps : int
        Max frames per second (applies only to event-pump rate).
    """

    def __init__(self, net, node_roles: dict[str, str], vehicle_ids: list[int], fps: int = 30) -> None:
        self._net = net
        self._node_roles = node_roles   # sumo_node_id → 'depot'|'customer'|'afs'
        self._vehicle_ids = vehicle_ids
        self._fps = fps

        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        # Build node position lookup and edge list once
        self._node_pos: dict[str, tuple[float, float]] = {}
        self._edges: list[tuple[str, str]] = []
        self._build_graph_geometry()

        # Compute transform parameters
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
            if edge.getID().startswith(":"):
                continue
            fn = edge.getFromNode().getID()
            tn = edge.getToNode().getID()
            self._edges.append((fn, tn))

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
            px = int(MARGIN + (x - min_x) * scale)
            py = int(MARGIN + (max_y - y) * scale)   # flip y
            return px, py

        return transform

    def _init_window(self) -> None:
        pygame.init()
        pygame.display.init()
        self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("EVRP Simulation")
        self._clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        sim_time_s: float,
        traffic_speed_ms: float,
        vehicle_positions: dict[int, str],     # vehicle_id → sumo_node_id
        served_customers: set[str],            # sumo_node_ids already served
        vehicle_states: dict[int, Any],        # vehicle_id → _VehicleState
    ) -> None:
        """Redraw the window. Call after every stop arrival."""
        if self._screen is None:
            self._init_window()

        # Pump events so the window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        surf = self._screen
        surf.fill(BG)

        self._draw_edges(surf)
        self._draw_nodes(surf, served_customers)
        self._draw_vehicles(surf, vehicle_positions)
        self._draw_panel(surf, sim_time_s, traffic_speed_ms, vehicle_positions, served_customers, vehicle_states)

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

    def _draw_edges(self, surf: pygame.Surface) -> None:
        for fn, tn in self._edges:
            if fn not in self._node_pos or tn not in self._node_pos:
                continue
            p1 = self._transform(*self._node_pos[fn])
            p2 = self._transform(*self._node_pos[tn])
            pygame.draw.line(surf, EDGE_COL, p1, p2, 2)

    def _draw_nodes(self, surf: pygame.Surface, served_customers: set[str]) -> None:
        font = pygame.font.Font(None, 16)
        for nid, pos in self._node_pos.items():
            role = self._node_roles.get(nid, "road")
            px, py = self._transform(*pos)

            if role == "depot":
                color = DEPOT_COL
                r = 10
            elif role == "customer":
                color = CUSTOMER_SERVED_COL if nid in served_customers else CUSTOMER_COL
                r = 8
            elif role == "afs":
                color = AFS_COL
                r = 9
            else:
                color = NODE_COL
                r = 5

            pygame.draw.circle(surf, color, (px, py), r)

            if role in ("depot", "customer", "afs"):
                label = {"depot": "D", "customer": "C", "afs": "F"}.get(role, "")
                txt = font.render(label, True, (255, 255, 255))
                surf.blit(txt, txt.get_rect(center=(px, py)))

    def _draw_vehicles(self, surf: pygame.Surface, vehicle_positions: dict[int, str]) -> None:
        font = pygame.font.Font(None, 18)
        for idx, (vid, nid) in enumerate(vehicle_positions.items()):
            if nid not in self._node_pos:
                continue
            color = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
            px, py = self._transform(*self._node_pos[nid])

            # Slightly offset multiple vehicles at same node
            offset = idx * 14
            px += offset

            pygame.draw.circle(surf, color, (px, py - 18), 9)
            txt = font.render(str(vid), True, (255, 255, 255))
            surf.blit(txt, txt.get_rect(center=(px, py - 18)))

    def _draw_panel(
        self,
        surf: pygame.Surface,
        sim_time_s: float,
        traffic_speed_ms: float,
        vehicle_positions: dict[int, str],
        served_customers: set[str],
        vehicle_states: dict[int, Any],
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
        line(f"Traffic: {traffic_speed_ms * 3.6:.1f} km/h")

        total_c = sum(1 for r in self._node_roles.values() if r == "customer")
        line(f"Customers: {len(served_customers)}/{total_c}")

        y += 10
        line("── Legend ──", color=TEXT_MID)
        self._legend_dot(surf, x, y, DEPOT_COL,            "Depot")        ; y += 22
        self._legend_dot(surf, x, y, CUSTOMER_COL,         "Customer (pending)") ; y += 22
        self._legend_dot(surf, x, y, CUSTOMER_SERVED_COL,  "Customer (served)")  ; y += 22
        self._legend_dot(surf, x, y, AFS_COL,              "AFS (fuel station)") ; y += 22
        y += 10

        line("── Vehicles ──", color=TEXT_MID)
        for idx, vid in enumerate(self._vehicle_ids):
            vs = vehicle_states.get(vid)
            color = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
            cur_node = vehicle_positions.get(vid, "?")
            self._legend_dot(surf, x, y, color, f"V{vid}  @{cur_node}") ; y += 22
            if vs is not None:
                line(f"  dist={vs.distance_m:.0f}m  fuel={vs.current_fuel:.2f}gal")
                line(f"  served={vs.customers_served}  recharges={vs.num_recharges}")

    @staticmethod
    def _legend_dot(surf: pygame.Surface, x: int, y: int, color: tuple, label: str) -> None:
        font = pygame.font.Font(None, 20)
        pygame.draw.circle(surf, color, (x + 7, y + 7), 7)
        surf.blit(font.render(label, True, TEXT_DARK), (x + 20, y))
