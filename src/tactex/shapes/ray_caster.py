import numpy as np


def ray_caster(
    ray_origin: np.ndarray, ray_goal_direction: np.ndarray, grid: np.ndarray
):
    """
    Cast rays from the given origins in the given directions and return the intersection points with the mesh.

    Args:
        mesh (trimesh.Trimesh): The mesh to cast rays against.
        ray_origins (np.ndarray): An array of shape (2,) representing the origins of the rays.
        ray_goal_directions (np.ndarray): An array of shape (2,) representing the goal direction of the ray. The actual direction vector is the difference between ray_goal_direction and ray_origin

    Returns:
        np.array: (x, y) coordinates of intersection with shape, or None if no intersetion occurred.
    """

    if np.array_equal(ray_origin, ray_goal_direction):
        raise ValueError("Ray origin and goal have to be different.")

    height, width = grid.shape

    miss_grid = np.zeros(grid.shape, dtype=bool)

    pixel_y, pixel_x = ray_origin

    origin = np.asarray(ray_origin, dtype=float) + 0.5  # center of the pixel
    goal = np.asarray(ray_goal_direction, dtype=float) + 0.5

    current_pos_y, current_pos_x = origin

    direction_vector = goal - origin
    dy, dx = direction_vector
    direction_x = int(np.sign(dx))
    direction_y = int(np.sign(dy))

    # Step sizes in x and y directions
    # delta_x = 1 / abs(dx) if dx != 0 else np.inf
    # delta_y = 1 / abs(dy) if dy != 0 else np.inf

    # Distance to next horizontal/vertical pixel wall

    # if direction_x > 0:
    #     distance_x = np.floor(current_pos_x) + 1 - current_pos_x
    #     distance_x = distance_x / dx
    # elif direction_x < 0:
    #     distance_x = current_pos_x - np.floor(current_pos_x)
    #     distance_x = distance_x / (-dx)
    # else:
    #     distance_x = np.inf

    # if direction_y > 0:
    #     distance_y = np.floor(current_pos_y) + 1 - current_pos_y
    #     distance_y = distance_y / dy
    # elif direction_y < 0:
    #     distance_y = current_pos_y - np.floor(current_pos_y)
    #     distance_y = distance_y / (-dy)
    # else:
    #     distance_y = np.inf

    while 0 <= pixel_x < width and 0 <= pixel_y < height:
        # Check occupancy
        if grid[pixel_y, pixel_x] == 1:
            hit = (pixel_y, pixel_x)
            return hit, miss_grid

        # Current pixel empty
        miss_grid[pixel_y, pixel_x] = True

        # Innerhalb der while-Schleife, nach der Belegungsprüfung:
        if direction_x > 0:
            distance_x = (pixel_x + 1 - current_pos_x) / dx
        elif direction_x < 0:
            distance_x = (pixel_x - current_pos_x) / dx
        else:
            distance_x = np.inf

        if direction_y > 0:
            distance_y = (pixel_y + 1 - current_pos_y) / dy
        elif direction_y < 0:
            distance_y = (pixel_y - current_pos_y) / dy
        else:
            distance_y = np.inf

        # Next pixel

        if distance_x > distance_y:
            pixel_y += direction_y
            delta_t = distance_y
        elif distance_x < distance_y:
            pixel_x += direction_x
            delta_t = distance_x
        else:  # distance_x == distance_y
            pixel_x += direction_x
            pixel_y += direction_y
            delta_t = distance_x

        current_pos_y += delta_t * dy
        current_pos_x += delta_t * dx

    return None, miss_grid
