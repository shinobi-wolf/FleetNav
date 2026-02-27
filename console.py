import pygame
import yaml
import sys
import math

from world_1.world import World

# --------------- Configuration ---------------

CONFIG_PATH = "config/world1.yaml"
EDGE_WEIGHT = 50
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
INFO_PANEL_HEIGHT = 140
FPS = 60

# Color map from World.default_colors strings to RGB
COLOR_MAP = {
    "red":     (220,  50,  50),
    "blue":    ( 50,  80, 220),
    "green":   ( 30, 180,  30),
    "yellow":  (220, 220,  30),
    "magenta": (220,  30, 220),
    "cyan":    ( 30, 220, 220),
    "orange":  (240, 160,  30),
    "black":   ( 60,  60,  60),
}

# Drone status colors
DRONE_IDLE_COLOR    = (100, 100, 100)
DRONE_TO_EV_COLOR   = (255, 200,  50)
DRONE_RETURN_COLOR  = (255, 130,  50)

# Misc colors
BG_COLOR          = (245, 245, 245)
WALL_COLOR        = ( 30,  30,  30)
GRID_COLOR        = (210, 210, 210)
WAREHOUSE_COLOR   = (255, 255, 255)
BASE_STATION_COLOR = (160, 160, 180)
TASK_COLOR        = (255,  80,  80)
TASK_DONE_COLOR   = (180, 220, 180)
INFO_BG_COLOR     = ( 35,  35,  45)
INFO_TEXT_COLOR    = (220, 220, 220)
PATH_ALPHA        = 120


# --------------- Helper functions ---------------

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cell_to_pixel(row, col, cell_size, offset_x, offset_y):
    """Convert 1-indexed maze (row, col) to pixel center."""
    px = offset_x + (col - 1) * cell_size + cell_size // 2
    py = offset_y + (row - 1) * cell_size + cell_size // 2
    return px, py


def cell_top_left(row, col, cell_size, offset_x, offset_y):
    """Top-left corner of a cell rectangle."""
    px = offset_x + (col - 1) * cell_size
    py = offset_y + (row - 1) * cell_size
    return px, py


def ev_color_rgb(world, idx):
    """Get RGB tuple for EV index."""
    color_idx = world.patrol_colors[idx] if idx < len(world.patrol_colors) else 0
    color_name = world.default_colors[color_idx % len(world.default_colors)]
    return COLOR_MAP.get(color_name, (200, 200, 200))


# --------------- Drawing ---------------

def draw_maze(surface, world, cell_size, ox, oy):
    """Draw maze grid and walls."""
    maze_size = world.maze_size

    # Light grid lines
    for r in range(maze_size + 1):
        y = oy + r * cell_size
        pygame.draw.line(surface, GRID_COLOR, (ox, y), (ox + maze_size * cell_size, y), 1)
    for c in range(maze_size + 1):
        x = ox + c * cell_size
        pygame.draw.line(surface, GRID_COLOR, (x, oy), (x, oy + maze_size * cell_size), 1)

    # Walls: iterate the dataframe
    wall_thickness = max(2, cell_size // 12)
    for _, row_data in world.df_maze.iterrows():
        r = int(row_data['row'])
        c = int(row_data['col'])
        tlx, tly = cell_top_left(r, c, cell_size, ox, oy)

        # North wall (top edge)
        if row_data['N'] == 0:
            pygame.draw.line(surface, WALL_COLOR,
                             (tlx, tly), (tlx + cell_size, tly), wall_thickness)
        # South wall (bottom edge)
        if row_data['S'] == 0:
            pygame.draw.line(surface, WALL_COLOR,
                             (tlx, tly + cell_size), (tlx + cell_size, tly + cell_size), wall_thickness)
        # West wall (left edge)
        if row_data['W'] == 0:
            pygame.draw.line(surface, WALL_COLOR,
                             (tlx, tly), (tlx, tly + cell_size), wall_thickness)
        # East wall (right edge)
        if row_data['E'] == 0:
            pygame.draw.line(surface, WALL_COLOR,
                             (tlx + cell_size, tly), (tlx + cell_size, tly + cell_size), wall_thickness)

    # Outer border
    pygame.draw.rect(surface, WALL_COLOR,
                     (ox, oy, maze_size * cell_size, maze_size * cell_size), wall_thickness)


def draw_warehouse(surface, world, cell_size, ox, oy, font):
    r, c = world.warehouse_pos
    cx, cy = cell_to_pixel(r, c, cell_size, ox, oy)
    half = cell_size // 2 - 2
    pygame.draw.rect(surface, WAREHOUSE_COLOR, (cx - half, cy - half, half * 2, half * 2))
    pygame.draw.rect(surface, (80, 80, 80), (cx - half, cy - half, half * 2, half * 2), 2)
    label = font.render("W", True, (0, 0, 0))
    surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))


def draw_base_stations(surface, world, cell_size, ox, oy, font):
    for idx, (r, c) in enumerate(world.base_stations):
        cx, cy = cell_to_pixel(r, c, cell_size, ox, oy)
        half = cell_size // 2 - 2
        pygame.draw.rect(surface, BASE_STATION_COLOR, (cx - half, cy - half, half * 2, half * 2))
        pygame.draw.rect(surface, (60, 60, 80), (cx - half, cy - half, half * 2, half * 2), 2)
        label = font.render(f"B{idx}", True, (20, 20, 40))
        surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))


def draw_tasks(surface, world, cell_size, ox, oy, font):
    """Draw all tasks. Completed ones are dimmed; remaining are bright."""
    # Figure out which tasks are still remaining
    remaining = set()
    for task_list in world.patrol_tasks:
        for t in task_list:
            remaining.add(t)
    # Also include active tasks that haven't been completed yet
    for t in world.active_tasks:
        if t is not None and t != world.warehouse_pos:
            remaining.add(t)

    radius = max(4, cell_size // 4)
    for task in world.all_tasks:
        cx, cy = cell_to_pixel(task[0], task[1], cell_size, ox, oy)
        if task in remaining:
            pygame.draw.circle(surface, TASK_COLOR, (cx, cy), radius)
            pygame.draw.circle(surface, (180, 40, 40), (cx, cy), radius, 2)
            label = font.render("T", True, (255, 255, 255))
            surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))
        else:
            # Completed task — green check
            pygame.draw.circle(surface, TASK_DONE_COLOR, (cx, cy), radius)
            pygame.draw.circle(surface, (100, 160, 100), (cx, cy), radius, 2)
            label = font.render("✓", True, (60, 140, 60))
            surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))


def draw_ev_paths(surface, world, state, cell_size, ox, oy):
    """Draw remaining path for each EV as a thin colored line."""
    for i, path in enumerate(state["patrol_paths"]):
        if not path:
            continue
        color = ev_color_rgb(world, i)
        # Fade color for path
        faded = tuple(min(255, c + 80) for c in color)
        points = [cell_to_pixel(world.patrol_positions[i][0], world.patrol_positions[i][1], cell_size, ox, oy)]
        for node in path:
            points.append(cell_to_pixel(node[0], node[1], cell_size, ox, oy))
        if len(points) >= 2:
            pygame.draw.lines(surface, faded, False, points, max(2, cell_size // 16))


def draw_evs(surface, world, state, cell_size, ox, oy, font):
    """Draw EV positions as colored circles with battery bars."""
    radius = max(6, cell_size // 3)
    for i, pos in enumerate(state["patrol_positions"]):
        cx, cy = cell_to_pixel(pos[0], pos[1], cell_size, ox, oy)
        color = ev_color_rgb(world, i)

        # Filled circle
        pygame.draw.circle(surface, color, (cx, cy), radius)
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), radius, 2)

        # EV label
        label = font.render(f"E{i}", True, (255, 255, 255))
        surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

        # Battery bar below EV
        bar_w = cell_size - 4
        bar_h = max(3, cell_size // 10)
        bar_x = cx - bar_w // 2
        bar_y = cy + radius + 2
        battery_pct = state["R_P"][i]

        # Background
        pygame.draw.rect(surface, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))
        # Fill
        if battery_pct > 0.5:
            bar_color = (50, 200, 50)
        elif battery_pct > 0.2:
            bar_color = (230, 200, 50)
        else:
            bar_color = (220, 50, 50)
        fill_w = int(bar_w * max(0, min(1, battery_pct)))
        if fill_w > 0:
            pygame.draw.rect(surface, bar_color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(surface, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h), 1)


def draw_drones(surface, world, state, cell_size, ox, oy, font_small):
    """Draw drone positions as small triangles/diamonds."""
    radius = max(4, cell_size // 5)
    for d_idx, pos in enumerate(state["drone_positions"]):
        # pos is [float, float] — continuous drone coordinates (row, col style)
        cx, cy = cell_to_pixel(pos[0], pos[1], cell_size, ox, oy)
        status = state["drone_status"][d_idx]

        if status == 0:
            color = DRONE_IDLE_COLOR
        elif status == 1:
            color = DRONE_TO_EV_COLOR
        else:
            color = DRONE_RETURN_COLOR

        # Draw diamond shape
        points = [
            (cx, cy - radius),
            (cx + radius, cy),
            (cx, cy + radius),
            (cx - radius, cy),
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (255, 255, 255), points, 1)

        # Tiny label
        label = font_small.render(f"D{d_idx}", True, (30, 30, 30))
        surface.blit(label, (cx - label.get_width() // 2, cy - radius - label.get_height() - 1))


def draw_info_panel(surface, world, state, frame, font, font_small):
    """Draw info panel at the bottom."""
    panel_y = WINDOW_HEIGHT - INFO_PANEL_HEIGHT
    pygame.draw.rect(surface, INFO_BG_COLOR, (0, panel_y, WINDOW_WIDTH, INFO_PANEL_HEIGHT))
    pygame.draw.line(surface, (80, 80, 100), (0, panel_y), (WINDOW_WIDTH, panel_y), 2)

    x_left = 15
    y = panel_y + 10
    line_h = 20

    # Row 1: Frame, tasks, completion
    completed = state["completed_tasks"]
    total = world.num_tasks
    flag = state["tasks_completed_flag"]
    status_text = "SIMULATION COMPLETE" if flag else "Running..."
    status_color = (100, 255, 100) if flag else (255, 220, 100)

    text = font.render(f"Frame: {frame}    Tasks: {completed}/{total}    Status: ", True, INFO_TEXT_COLOR)
    surface.blit(text, (x_left, y))
    status_label = font.render(status_text, True, status_color)
    surface.blit(status_label, (x_left + text.get_width(), y))

    y += line_h

    # Row 2: Distances
    ev_dist = state["ev_distance_traveled"]
    drone_dist = state["drone_distance_traveled"]
    text2 = font.render(f"EV Distance: {ev_dist:.1f}    Drone Distance: {drone_dist:.1f}    BMS: {'ON' if world.bms else 'OFF'}", True, INFO_TEXT_COLOR)
    surface.blit(text2, (x_left, y))

    y += line_h

    # Row 3: Per-EV battery
    batteries = []
    for i in range(state["num_patrols"]):
        pct = state["R_P"][i] * 100
        color = ev_color_rgb(world, i)
        batteries.append((f"EV{i}: {pct:.0f}%", color))

    bx = x_left
    for label_text, color in batteries:
        label = font_small.render(label_text, True, color)
        surface.blit(label, (bx, y))
        bx += label.get_width() + 20

    y += line_h

    # Row 4: Instructions
    instr = font_small.render("Press RETURN to step  |  SPACE for auto-run  |  ESC to quit", True, (150, 150, 170))
    surface.blit(instr, (x_left, y))


def draw_legend(surface, world, font_small, ox, oy):
    """Draw a small legend in the top-right corner."""
    items = [
        ("W  Warehouse", WAREHOUSE_COLOR, (80, 80, 80)),
        ("B  Base Station", BASE_STATION_COLOR, (60, 60, 80)),
        ("T  Task", TASK_COLOR, (180, 40, 40)),
        ("✓  Completed", TASK_DONE_COLOR, (100, 160, 100)),
        ("◆  Drone", DRONE_TO_EV_COLOR, (200, 150, 30)),
    ]
    lx = WINDOW_WIDTH - 160
    ly = oy + 5
    for text, bg_color, border_color in items:
        pygame.draw.rect(surface, bg_color, (lx, ly, 12, 12))
        pygame.draw.rect(surface, border_color, (lx, ly, 12, 12), 1)
        label = font_small.render(text, True, (60, 60, 60))
        surface.blit(label, (lx + 16, ly - 1))
        ly += 18


# --------------- Main draw function ---------------

def draw_all(surface, world, state, frame, cell_size, ox, oy, font, font_small):
    surface.fill(BG_COLOR)
    draw_maze(surface, world, cell_size, ox, oy)
    draw_warehouse(surface, world, cell_size, ox, oy, font_small)
    draw_base_stations(surface, world, cell_size, ox, oy, font_small)
    draw_tasks(surface, world, cell_size, ox, oy, font_small)
    draw_ev_paths(surface, world, state, cell_size, ox, oy)
    draw_evs(surface, world, state, cell_size, ox, oy, font_small)
    draw_drones(surface, world, state, cell_size, ox, oy, font_small)
    draw_info_panel(surface, world, state, frame, font, font_small)
    draw_legend(surface, world, font_small, ox, oy)
    pygame.display.flip()


# --------------- Entry point ---------------

def startup():
    """
    Main entry point: loads config, creates World, opens a pygame window,
    and runs an interactive step-by-step simulation.
    """
    # Load config
    config = load_config(CONFIG_PATH)

    # Create and initialize world
    world = World(config)
    world.initialize(EDGE_WEIGHT)
    state = world.get_world_state()

    # Pygame setup
    pygame.init()
    surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("DroneSim — EVRP Visualization")
    clock = pygame.time.Clock()

    # Fonts
    font = pygame.font.SysFont("monospace", 16, bold=True)
    font_small = pygame.font.SysFont("monospace", 13)

    # Compute cell size to fit the maze in the available area
    maze_area_h = WINDOW_HEIGHT - INFO_PANEL_HEIGHT - 20
    maze_area_w = WINDOW_WIDTH - 20
    cell_size = min(maze_area_w, maze_area_h) // world.maze_size
    # Center the maze
    total_maze_w = cell_size * world.maze_size
    total_maze_h = cell_size * world.maze_size
    ox = (WINDOW_WIDTH - total_maze_w) // 2
    oy = (WINDOW_HEIGHT - INFO_PANEL_HEIGHT - total_maze_h) // 2

    frame = 0
    auto_run = False
    auto_delay = 200  # ms between auto steps
    last_auto_step = 0

    running = True
    draw_all(surface, world, state, frame, cell_size, ox, oy, font, font_small)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_RETURN:
                    if not state.get("tasks_completed_flag", False):
                        world.simulate(timesteps=1)
                        state = world.get_world_state()
                        frame += 1
                    draw_all(surface, world, state, frame, cell_size, ox, oy, font, font_small)

                elif event.key == pygame.K_SPACE:
                    auto_run = not auto_run
                    last_auto_step = pygame.time.get_ticks()

                elif event.key == pygame.K_UP:
                    auto_delay = max(20, auto_delay - 50)
                elif event.key == pygame.K_DOWN:
                    auto_delay = min(2000, auto_delay + 50)

        # Auto-run mode
        if auto_run and not state.get("tasks_completed_flag", False):
            now = pygame.time.get_ticks()
            if now - last_auto_step >= auto_delay:
                world.simulate(timesteps=1)
                state = world.get_world_state()
                frame += 1
                last_auto_step = now
                draw_all(surface, world, state, frame, cell_size, ox, oy, font, font_small)

        clock.tick(FPS)

    pygame.quit()
    sys.exit()
