from argparse import ArgumentParser, Namespace
from collections.abc import Callable

import numpy as np

from tactex.shapes.generators import circle, rectangle, triangle
from tactex.shapes.ioop import save

GENERATOR_MODULE_REGISTRY: dict[
    str, Callable[[int, int, np.random.Generator], np.ndarray]
] = {
    "circle": circle,
    "rectangle": rectangle,
    "triangle": triangle,
}


def main(args: Namespace):
    rng: np.random.Generator = np.random.default_rng(seed=args.seed)
    for shape in args.shapes:
        print(
            f"Generating {args.number_of_samples} datapoints for shape: '{shape}' with resolution {args.resolution}x{args.resolution}."
        )
        occupancy_grids, sdf_grids, params = GENERATOR_MODULE_REGISTRY[shape].generate(
            n=args.number_of_samples, resolution=args.resolution, rng=rng
        )
        save(
            shape=shape,
            split=args.split,
            occupancy_grids=occupancy_grids,
            sdf_grids=sdf_grids,
            resolution=args.resolution,
            seed=args.seed,
            **params,
        )


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        "--shapes",
        type=str,
        choices=list(GENERATOR_MODULE_REGISTRY),
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--resolution", type=int, choices=[16, 32, 64, 128, 256, 512], required=True
    )
    parser.add_argument("--number-of-samples", type=int, required=True)
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)

    args: Namespace = parser.parse_args()

    main(args=args)
