import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pygame
import os

# Define colors for pygame
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
DARK_GREEN = (0, 100, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

G = nx.read_gml('data/maze.gml')
G_converted = nx.Graph()
for u, v, data in G.edges(data=True):
    u_converted = eval(u) if isinstance(u, str) else u
    v_converted = eval(v) if isinstance(v, str) else v
    G_converted.add_edge(u_converted, v_converted, **data)
G = G_converted

yellow_roads = []
red_roads = []
for u, v, data in G.edges(data=True):
    if 'traffic' in data:
        if data['traffic'] == 1:
            red_roads.append((u, v))
        elif data['traffic'] == 2:
            yellow_roads.append((u, v))

pygame.init()
pygame.display.set_caption("LMD Simulation")
pygame.display.set_mode((1, 1), pygame.NOFRAME)  # Create a minimal display to enable image loading
clock = pygame.time.Clock()

# Calculate screen dimensions based on maze size and cell pixel size
# Add padding for potential info text display later
padding = 200
cell_size_px = 70
max_row, max_col = 7, 7
wall_thickness = 2
warehouse_pos = (max_row//2+1, max_col//2+1)
active_task = (1,5)
ugv_pos = (3,3)
ugv_heading_dir = 1
time_elapsed = 0
current_timestep = 0
total_ev_distance = 0

screen_width = max_col * cell_size_px
screen_height = max_row * cell_size_px + padding # Add padding at the bottom

# Load font
try:
    font = pygame.font.SysFont(None, 24) # Adjust font size as needed
except Exception as e:
    print(f"Warning: Could not load system font. Text rendering might fail. Error: {e}")
    font = pygame.font.Font(None, 24) # Fallback to default font

truck_img_path = os.path.join('data/truck.png')

def _cell_to_pygame(row, col):
        """Converts maze (row, col) to pygame pixel coordinates (top-left of cell)."""
        # Pygame origin (0,0) is top-left
        px = (col - 1) * cell_size_px
        py = (row - 1) * cell_size_px # Adjusted for top-left origin
        return int(px), int(py)

def _cell_to_pygame_center(row, col):
        """Converts maze (row, col) to pygame pixel coordinates (center of cell)."""
        px, py = _cell_to_pygame(row, col)
        px += cell_size_px // 2
        py += cell_size_px // 2
        return int(px), int(py)

surface = pygame.Surface((screen_width, screen_height))
render_surface = surface

truck_image = pygame.image.load(truck_img_path).convert_alpha() # Use convert_alpha for transparency
scale_factor = 0.8
img_size = int(cell_size_px * scale_factor)
truck_image = pygame.transform.scale(truck_image, (img_size, img_size))

# --- Drawing ---
render_surface.fill(WHITE) # Clear surface

# --- Draw Maze Area (Grid and Walls) ---
maze_area_height = max_row * cell_size_px
# --- Draw Maze Area (Grid and Walls) using self.G ---
for r in range(1, max_row + 1):
    for c in range(1, max_col + 1):
        current_node = (r, c)
        # Skip if node somehow doesn't exist in the graph (should not happen for grid)
        if current_node not in G:
            continue

        px_top_left, py_top_left = _cell_to_pygame(r, c)

        # Check North Wall: Draw wall if no edge exists between current node and North neighbor (r-1, c)
        # G.has_edge handles boundary cases (e.g., r=1, neighbor (0,c) doesn't exist, so has_edge is False)
        north_neighbor = (r - 1, c)
        if not G.has_edge(current_node, north_neighbor):
            # Draw the North wall of cell (r, c)
            pygame.draw.line(render_surface, BLACK, (px_top_left, py_top_left), (px_top_left + cell_size_px, py_top_left), wall_thickness)
        else:
            pygame.draw.line(render_surface, GRAY, (px_top_left, py_top_left), (px_top_left + cell_size_px, py_top_left), wall_thickness)

        # Check West Wall: Draw wall if no edge exists between current node and West neighbor (r, c-1)
        west_neighbor = (r, c - 1)
        if not G.has_edge(current_node, west_neighbor):
            # Draw the West wall of cell (r, c)
            pygame.draw.line(render_surface, BLACK, (px_top_left, py_top_left), (px_top_left, py_top_left + cell_size_px), wall_thickness)
        else:
            pygame.draw.line(render_surface, GRAY, (px_top_left, py_top_left), (px_top_left, py_top_left + cell_size_px), wall_thickness)

# Draw outer boundary walls (South and East edges explicitly)
# This ensures the bottom border of the last row and the right border of the last column are drawn.
pygame.draw.line(render_surface, BLACK, (0, maze_area_height), (screen_width, maze_area_height), wall_thickness) # Bottom boundary
pygame.draw.line(render_surface, BLACK, (screen_width, 0), (screen_width, maze_area_height), wall_thickness) # Right boundary


# --- Draw Traffic Roads ---
road_thickness = max(1, int(cell_size_px * 0.15)) # Adjust thickness relative to cell size
for u, v in red_roads:
    u_px_c, u_py_c = _cell_to_pygame_center(u[0], u[1])
    v_px_c, v_py_c = _cell_to_pygame_center(v[0], v[1])
    pygame.draw.line(render_surface, RED, (u_px_c, u_py_c), (v_px_c, v_py_c), road_thickness)
for u, v in yellow_roads:
    u_px_c, u_py_c = _cell_to_pygame_center(u[0], u[1])
    v_px_c, v_py_c = _cell_to_pygame_center(v[0], v[1])
    pygame.draw.line(render_surface, YELLOW, (u_px_c, u_py_c), (v_px_c, v_py_c), road_thickness)


# --- Draw Warehouse ---
wh_r, wh_c = warehouse_pos
wh_px_c, wh_py_c = _cell_to_pygame_center(wh_r, wh_c)
warehouse_radius = int(cell_size_px * 0.4)
pygame.draw.circle(render_surface, BLUE, (wh_px_c, wh_py_c), warehouse_radius)

# --- Draw Tasks ---
task_r, task_c = active_task
task_px_c, task_py_c = _cell_to_pygame_center(task_r, task_c)
task_radius = int(cell_size_px * 0.25)
pygame.draw.circle(render_surface, MAGENTA, (task_px_c, task_py_c), task_radius)

ugv_r, ugv_c = ugv_pos
ugv_px_c, ugv_py_c = _cell_to_pygame_center(ugv_r, ugv_c)

# Draw UGV (Image or Shape)
truck_img_rotated = pygame.transform.rotate(truck_image, (ugv_heading_dir-1)%4 * -90) # Rotate image based on heading

img_rect = truck_img_rotated.get_rect(center=(ugv_px_c, ugv_py_c))
render_surface.blit(truck_img_rotated, img_rect)

# For rgb_array mode, convert surface to numpy array and return

# --- Draw Info Text ---
if font:
    info_y_start = maze_area_height + 5 # Start below maze area
    info_line_height = 20
    texts = [
        f"Time Elapsed: {time_elapsed:.2f}s",
        f"Timestep: {current_timestep}",
        f"Total Dist: {total_ev_distance:.1f}m",
    ]
    for i, text in enumerate(texts):
        text_surface = font.render(text, True, BLACK)
        render_surface.blit(text_surface, (5, info_y_start + i * info_line_height))

# # For human mode, update the display
# pygame.display.flip()

# Save the rendered maze as a PNG image
output_path = 'maze_output.png'
pygame.image.save(render_surface, output_path)
print(f"Maze image saved to {output_path}")