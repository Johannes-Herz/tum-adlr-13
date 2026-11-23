import numpy as np

from tactex.timings import timed
from tactex.typing import OccupancyGrid2D, Ray2D, Traversal2D, TraversalGrid2D


@timed()
def sample(resolution: int, b: int, n: int, rng: np.random.Generator) -> Ray2D:
    """Sample random 2D rays between points on the grid's circumcircle (float32).

    Same semantics as `tactex.sensing.raycasting.sample`, but avoids float64
    intermediates and redundant `np.stack` allocations.
    """
    angles: np.ndarray = rng.random((b, n, 2), dtype=np.float32) * np.float32(
        2.0 * np.pi
    )
    radius: np.float32 = np.float32(resolution * 0.5 * np.sqrt(2.0))
    center: np.float32 = np.float32(resolution * 0.5)
    rays: Ray2D = np.empty((b, n, 2, 2), dtype=np.float32)
    np.cos(angles, out=rays[..., 0])
    np.sin(angles, out=rays[..., 1])
    rays *= radius
    rays += center
    return rays


@timed()
def cast(
    occupancy_grids: OccupancyGrid2D, traversals: Traversal2D, num_rays: np.ndarray
) -> TraversalGrid2D:
    """Optimized drop-in replacement for `tactex.sensing.raycasting.cast`.

    Produces bit-for-bit identical output to the original `cast`, while:
    - Replacing two `int64` `np.cumsum` calls with `argmax` + `hit_limit`
    - Using 1D flat indexing instead of 3D fancy indexing with `int64` `batch_idx`
    - Using a zero-copy `uint32` view for `[0, H)` / `[0, W)` bounds checks
    """
    B, H, W = occupancy_grids.shape
    _, N, M, _ = traversals.shape

    active_rays: np.ndarray = np.arange(N) < num_rays.reshape(B, 1)  # (B, N)
    tx: np.ndarray = traversals[..., 0]
    ty: np.ndarray = traversals[..., 1]
    in_bound_mask: np.ndarray = (tx.view(np.uint32) < H) & (ty.view(np.uint32) < W)

    cx: np.ndarray = np.clip(tx, 0, H - 1)
    cy: np.ndarray = np.clip(ty, 0, W - 1)
    batch_offsets: np.ndarray = (np.arange(B, dtype=np.int32) * (H * W))[:, None, None]
    flat_idx: np.ndarray = batch_offsets + cx * W + cy  # (B, N, M) int32

    occupancies: np.ndarray = occupancy_grids.astype(np.bool_, copy=False).reshape(-1)[
        flat_idx
    ]
    occupancies &= in_bound_mask

    # First hit along each ray (argmax returns 0 if all False; verify with has_hit)
    first_hit: np.ndarray = np.argmax(occupancies, axis=-1)  # (B, N)
    b_grid: np.ndarray = np.arange(B)[:, None]
    n_grid: np.ndarray = np.arange(N)[None, :]
    has_hit: np.ndarray = occupancies[b_grid, n_grid, first_hit]

    # Occupied: at most 1 cell per active ray
    occ_flat: np.ndarray = np.zeros(B * H * W, dtype=np.bool_)
    occ_b, occ_n = np.nonzero(has_hit & active_rays)
    occ_m: np.ndarray = first_hit[occ_b, occ_n]
    occ_flat[flat_idx[occ_b, occ_n, occ_m]] = True

    # Free: all in-bound active steps strictly before the first hit
    hit_limit: np.ndarray = np.where(has_hit, first_hit, M).astype(np.int32)
    step_idx: np.ndarray = np.arange(M, dtype=np.int32)
    free_mask: np.ndarray = (
        (step_idx < hit_limit[..., None]) & in_bound_mask & active_rays[..., None]
    )
    free_flat: np.ndarray = np.zeros(B * H * W, dtype=np.bool_)
    free_flat[flat_idx[free_mask]] = True

    occ: np.ndarray = occ_flat.reshape(B, H, W)
    free: np.ndarray = free_flat.reshape(B, H, W)
    out: TraversalGrid2D = np.empty((B, 3, H, W), dtype=np.bool_)
    out[:, 0] = occ
    out[:, 1] = free
    np.logical_not(occ | free, out=out[:, 2])
    return out


def _dda_cast_flat(
    occupancy_grids: OccupancyGrid2D,
    x0: np.ndarray,
    y0: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    batch_idx: np.ndarray,
) -> TraversalGrid2D:
    """Core vectorized 2D DDA (Amanatides & Woo) over a flat list of active rays.

    Each ray is slab-clipped to `[0, H) x [0, W)` and stepped in lockstep across
    all active rays, compacting out rays as soon as they hit an occupied cell or
    leave the grid.
    """
    B, H, W = occupancy_grids.shape
    if x0.size == 0:
        out = np.zeros((B, 3, H, W), dtype=np.bool_)
        out[:, 2] = True
        return out

    # 1. Slab-clip rays to [0, H) x [0, W)
    eps = np.float32(1e-12)
    dx_nonzero = np.abs(dx) > eps
    dy_nonzero = np.abs(dy) > eps
    inv_dx = np.where(
        dx_nonzero,
        np.float32(1.0) / np.where(dx_nonzero, dx, np.float32(1.0)),
        np.float32(1e30),
    )
    inv_dy = np.where(
        dy_nonzero,
        np.float32(1.0) / np.where(dy_nonzero, dy, np.float32(1.0)),
        np.float32(1e30),
    )

    tx0 = (np.float32(0.0) - x0) * inv_dx
    tx1 = (np.float32(H) - x0) * inv_dx
    tmin_x = np.minimum(tx0, tx1)
    tmax_x = np.maximum(tx0, tx1)

    ty0 = (np.float32(0.0) - y0) * inv_dy
    ty1 = (np.float32(W) - y0) * inv_dy
    tmin_y = np.minimum(ty0, ty1)
    tmax_y = np.maximum(ty0, ty1)

    t_enter = np.maximum(np.maximum(tmin_x, tmin_y), np.float32(0.0))
    t_exit = np.minimum(np.minimum(tmax_x, tmax_y), np.float32(1.0))

    valid = t_enter < t_exit
    if not np.any(valid):
        out = np.zeros((B, 3, H, W), dtype=np.bool_)
        out[:, 2] = True
        return out

    # Drop rays that never intersect the grid
    x0 = x0[valid]
    y0 = y0[valid]
    dx = dx[valid]
    dy = dy[valid]
    inv_dx = inv_dx[valid]
    inv_dy = inv_dy[valid]
    t_enter = t_enter[valid]
    t_exit = t_exit[valid]
    batch_idx = batch_idx[valid]

    # Nudge slightly past entry boundary so floor() lands inside [0, H) x [0, W)
    t_start = np.where(
        t_enter > np.float32(0.0),
        np.minimum(t_enter + np.float32(1e-5), np.float32(0.5) * (t_enter + t_exit)),
        np.float32(0.0),
    )
    px = x0 + t_start * dx
    py = y0 + t_start * dy

    ix = np.clip(np.floor(px).astype(np.int32), 0, H - 1)
    iy = np.clip(np.floor(py).astype(np.int32), 0, W - 1)

    step_x = np.where(dx >= 0, 1, -1).astype(np.int32)
    step_y = np.where(dy >= 0, 1, -1).astype(np.int32)

    next_x = ix + (step_x > 0)
    next_y = iy + (step_y > 0)

    t_max_x = np.where(np.abs(dx) > eps, (next_x - px) * inv_dx, np.float32(1e30))
    t_max_y = np.where(np.abs(dy) > eps, (next_y - py) * inv_dy, np.float32(1e30))
    t_delta_x = np.abs(inv_dx)
    t_delta_y = np.abs(inv_dy)

    grid_flat = occupancy_grids.astype(np.bool_, copy=False).reshape(-1)
    occ_flat = np.zeros(B * H * W, dtype=np.bool_)
    free_flat = np.zeros(B * H * W, dtype=np.bool_)
    b_off = batch_idx * (H * W)

    # 2. Vectorized DDA loop with compaction (all active arrays stay in L1/L2 cache)
    while ix.size > 0:
        flat_pos = b_off + ix * W + iy
        hit = grid_flat[flat_pos]
        if np.any(hit):
            occ_flat[flat_pos[hit]] = True

        alive = ~hit
        if not np.any(alive):
            break
        free_flat[flat_pos[alive]] = True

        if not np.all(alive):
            ix = ix[alive]
            iy = iy[alive]
            b_off = b_off[alive]
            step_x = step_x[alive]
            step_y = step_y[alive]
            t_max_x = t_max_x[alive]
            t_max_y = t_max_y[alive]
            t_delta_x = t_delta_x[alive]
            t_delta_y = t_delta_y[alive]

        go_x = t_max_x < t_max_y
        ix = np.where(go_x, ix + step_x, ix)
        t_max_x = np.where(go_x, t_max_x + t_delta_x, t_max_x)
        iy = np.where(go_x, iy, iy + step_y)
        t_max_y = np.where(go_x, t_max_y, t_max_y + t_delta_y)

        in_bounds = (ix.view(np.uint32) < H) & (iy.view(np.uint32) < W)
        if not np.all(in_bounds):
            ix = ix[in_bounds]
            iy = iy[in_bounds]
            b_off = b_off[in_bounds]
            step_x = step_x[in_bounds]
            step_y = step_y[in_bounds]
            t_max_x = t_max_x[in_bounds]
            t_max_y = t_max_y[in_bounds]
            t_delta_x = t_delta_x[in_bounds]
            t_delta_y = t_delta_y[in_bounds]

    occ = occ_flat.reshape(B, H, W)
    free = free_flat.reshape(B, H, W)
    out = np.empty((B, 3, H, W), dtype=np.bool_)
    out[:, 0] = occ
    out[:, 1] = free
    np.logical_not(occ | free, out=out[:, 2])
    return out


@timed()
def cast_rays_dda(
    occupancy_grids: OccupancyGrid2D,
    rays: Ray2D,
    num_rays: np.ndarray | None = None,
) -> TraversalGrid2D:
    """Cast continuous 2D rays onto occupancy grids via vectorized 2D DDA.

    Replaces `traverse(rays, res)` + `cast(grids, traversals, num_rays)` in a
    single pass without materializing `(B, N, M, 2)` traversal arrays.
    """
    B, _, _ = occupancy_grids.shape
    _, N, _, _ = rays.shape
    if num_rays is not None:
        active = np.arange(N) < num_rays.reshape(B, 1)
        b_idx, n_idx = np.nonzero(active)
        x0 = rays[b_idx, n_idx, 0, 0].astype(np.float32, copy=False)
        y0 = rays[b_idx, n_idx, 0, 1].astype(np.float32, copy=False)
        dx = rays[b_idx, n_idx, 1, 0].astype(np.float32, copy=False) - x0
        dy = rays[b_idx, n_idx, 1, 1].astype(np.float32, copy=False) - y0
        batch_idx = b_idx.astype(np.int32)
    else:
        x0 = rays[..., 0, 0].astype(np.float32, copy=False).ravel()
        y0 = rays[..., 0, 1].astype(np.float32, copy=False).ravel()
        dx = rays[..., 1, 0].astype(np.float32, copy=False).ravel() - x0
        dy = rays[..., 1, 1].astype(np.float32, copy=False).ravel() - y0
        batch_idx = np.repeat(np.arange(B, dtype=np.int32), N)

    return _dda_cast_flat(occupancy_grids, x0, y0, dx, dy, batch_idx)


@timed()
def sample_and_cast_dda(
    occupancy_grids: OccupancyGrid2D,
    num_rays: np.ndarray,
    rng: np.random.Generator,
) -> TraversalGrid2D:
    """Fused ray sampling + vectorized 2D DDA casting for `TactileCollate`.

    Only samples the exact `sum(num_rays)` active rays (avoiding ~50% wasted
    inactive rays from padding to `max(num_rays)`), then runs vectorized DDA.
    """
    B, H, W = occupancy_grids.shape
    counts = num_rays.ravel()
    total_rays = int(counts.sum())
    if total_rays == 0:
        out = np.zeros((B, 3, H, W), dtype=np.bool_)
        out[:, 2] = True
        return out

    angles = rng.random((total_rays, 2), dtype=np.float32) * np.float32(2.0 * np.pi)
    radius = np.float32(H * 0.5 * np.sqrt(2.0))
    cx = np.float32(H * 0.5)
    cy = np.float32(W * 0.5)

    x0 = radius * np.cos(angles[:, 0]) + cx
    y0 = radius * np.sin(angles[:, 0]) + cy
    dx = (radius * np.cos(angles[:, 1]) + cx) - x0
    dy = (radius * np.sin(angles[:, 1]) + cy) - y0
    batch_idx = np.repeat(np.arange(B, dtype=np.int32), counts)

    return _dda_cast_flat(occupancy_grids, x0, y0, dx, dy, batch_idx)
