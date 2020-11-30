import pathlib as pt
import pymethods as pma
import multiprocessing as mp
import CGALMethods as cm
import numpy as np
import click
import logging
import pyvista as pv
from scipy.spatial import cKDTree
from typing import List

logger = logging.getLogger()

def string_to_list_ints(data):
    try:
        numbers = data.strip("()")
        numbers = [int(number) for number in data.split(",")]
        return numbers
    except:
        return data

def create_structured_grid_over_cylinder(
    folder        : pt.Path,
    unwrapped_path: str = "unwrap_data.npz",
    save_filename : str = "structured_volume.npy",
    dimension    : List[float] = (16, 16, 384),
    log_info      : bool = True
):
    logger.propagate = log_info
    logger.info("Processing %s"%folder.name)

    unwrapped_data = np.load(folder/unwrapped_path)

    transfinite_interpolator = \
        pma.algorithms.transfinite_interpolation.TransfiniteCylinder(
            pma.arrays.structured.CylindricalSurface(unwrapped_data["points_grid"])
        )
    structured_mesh_internal = transfinite_interpolator.pts_mesh_uniform(
        *dimension)

    np.save(
        folder/save_filename,
        structured_mesh_internal,
    )

@click.command()
@click.option("--main_folder", type=pt.Path, default=".")
@click.option("--n_processors", default=8, type=int)
@click.option("--unwrapped_path", default="unwrap_data.npz", type=str)
@click.option("--dimension", default=(16, 16, 384), type=string_to_list_ints)
@click.option("--save_filename", default="structured_volume.npy", type=str)
@click.option("--log_info", default=bool)
def main(
    main_folder, n_processors, unwrapped_path, dimension, save_filename, log_info
):
    logger.propagate = log_info
    # NOTE: moify this method to locate the case folders
    all_folders_to_process = [
        folder for folder in main_folder.glob("*") if folder.is_dir()]
    logger.info("Starting Unwrapping For %d Items"%len(all_folders_to_process))
    if n_processors == 1:
        [
            create_structured_grid_over_cylinder(folder) for
            folder in all_folders_to_process
        ]
    else:
        with mp.Pool(n_processors) as p:
            resultsList = [
                p.apply_async(
                    create_structured_grid_over_cylinder, (folder, ),
                    kwds=dict(
                        unwrapped_path = unwrapped_path,
                        save_filename  = save_filename,
                        dimension      = dimension,
                        log_info       = log_info
                    )
                )
                for folder in all_folders_to_process
            ]
            for result in resultsList:
                result.get()

    logger.info("Completed Structured Volume Generation For All")

if __name__ == "__main__":
    main()
