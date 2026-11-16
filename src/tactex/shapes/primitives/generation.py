import numpy as np

from tactex.shapes.primitives.config import GridConfig, PrimitiveConfig
from tactex.shapes.primitives.grid import Grid2D, rasterize, sdf_to_occupancy
from tactex.shapes.primitives.sampling import SampleFunction, SampleParams
from tactex.shapes.primitives.sdfs import (
    SignedDistanceFunction,
    SignedDistanceFunctionBuilder,
)


def generate(
    primitive_config: PrimitiveConfig,
    grid_config: GridConfig,
    sdf_builder: SignedDistanceFunctionBuilder,
    sample_fn: SampleFunction,
    rng: np.random.Generator,
) -> tuple[Grid2D, Grid2D, SampleParams]:
    sample_params: SampleParams = sample_fn(
        primitive_config=primitive_config, grid_config=grid_config, rng=rng
    )
    sdf: SignedDistanceFunction = sdf_builder(**sample_params)
    sdf_grids: Grid2D = rasterize(sdf=sdf, grid_config=grid_config)
    occupancy_grids: Grid2D = sdf_to_occupancy(sdf_grids=sdf_grids)
    return occupancy_grids, sdf_grids, sample_params
