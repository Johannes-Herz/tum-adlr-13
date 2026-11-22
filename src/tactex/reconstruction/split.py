import numpy as np


def split_indices(
    grid_count: int,
    samples_per_object: int,
    num_objects: int,
    val_fraction: float,
    test_fraction: float,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Splits datset in test, validation, and train according to their objects.
    """
    if samples_per_object < 1 or grid_count % samples_per_object != 0:
        raise ValueError("Grid count must be devisible by samples per object.")
    if not (0 < val_fraction < 1 and 0 <= test_fraction < 1 - val_fraction):
        raise ValueError("Require val > 0, test >= 0 and val + test < 1.")
    if grid_count != samples_per_object * num_objects:
        raise ValueError(
            "Number of objects does not match grid count and samples per object."
        )

    objects = np.random.default_rng(seed).permutation(num_objects)

    n_val = int(len(objects) * val_fraction)
    n_test = int(len(objects) * test_fraction)
    n_train = len(objects) - n_val - n_test

    if n_val == 0 or (test_fraction > 0 and n_test == 0) or n_train == 0:
        raise ValueError("Too few objects for nonempty train/validation/test splits.")

    object_splits = {
        "train": objects[:n_train],
        "val": objects[n_train : n_train + n_val],
        "test": objects[n_train + n_val :],
    }

    grid_splits = {
        split_name: (
            object_ids[:, None] * samples_per_object + np.arange(samples_per_object)
        ).reshape(-1)
        for split_name, object_ids in object_splits.items()
    }

    return grid_splits
