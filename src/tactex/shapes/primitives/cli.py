from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np

from tactex.shapes import ioop
from tactex.shapes.primitives import (
    GenerationConfig,
    SampleFunction,
    SignedDistanceFunctionBuilder,
    generation,
    sampling,
    sdfs,
)


def main(generation_config: GenerationConfig) -> dict[str, dict[str, Any]]:
    rng: np.random.Generator = np.random.default_rng(seed=generation_config.seed)

    generations: dict[str, dict[str, Any]] = {}
    for primitive_config in generation_config.primitive_configs:
        sample_fn: SampleFunction = sampling.REGISTRY[primitive_config.primitive]
        sdf_builder: SignedDistanceFunctionBuilder = sdfs.REGISTRY[
            primitive_config.primitive
        ]
        occupancy_grids, sdf_grids, sample_params = generation.generate(
            primitive_config=primitive_config,
            grid_config=generation_config.grid_config,
            sdf_builder=sdf_builder,
            sample_fn=sample_fn,
            rng=rng,
        )
        path: Path = ioop.save(
            generator=primitive_config.primitive,
            occupancy_grids=occupancy_grids,
            sdf_grids=sdf_grids,
            resolution=generation_config.grid_config.resolution,
            seed=generation_config.seed,
            generator_params={},
            sample_params=sample_params,
        )
        generations[primitive_config.primitive] = {
            "path": path,
            "occupancy_grids": occupancy_grids,
            "sdf_grids": sdf_grids,
            "sample_params": sample_params,
        }
    return generations


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        "--config", type=Path, required=False, help="Path to a YAML config file."
    )

    args: Namespace = parser.parse_args()

    generation_config: GenerationConfig = (
        GenerationConfig.from_yaml(args.config) if args.config else GenerationConfig()
    )
    main(generation_config=generation_config)
