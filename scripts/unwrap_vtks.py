import pathlib as pt
import pymethods as pma
import multiprocessing as mp
import CGALMethods as cm
import numpy as np
import click
import logging
import pyvista as pv

'''
    The following script unwraps all cases withn a folder.
'''

logger = logging.getLogger()

def unwrap_vtk_and_save(
    folder         : pt.Path,
    centreline_path: str = "centreline.npy",
    wall_path      : str = "WALL/WALL_400.vtk",
    save_filename  : str = "unwrap_data.npz",
    log_info       : bool = True
):
    logger.propagate = log_info
    logger.info("Processing: %s"%folder.name)

    wall_data  = pv.read(folder/wall_path)
    centreline = np.load(folder/centreline_path)

    points_grid, fields_grids = \
        pma.algorithms.unwrapping.unwrap_cylinder_vtk_from_centerline(
            pma.arrays.Curve(centreline), wall_data
    )

    np.savez(
        folder/save_filename,
        points_grid=points_grid,
        fields_grd=fields_grids
    )

@click.command()
@click.option("--main_folder", type=pt.Path, default=".")
@click.option("--n_processors", default=8, type=int)
@click.option("--wall_path", default="WALL/WALL_400.vtk", type=str)
@click.option("--centreline_path", default="centreline.npy", type=str)
@click.option("--save_filename", default="unwrap_data.npz", type=str)
@click.option("--log_info", default=bool)
def main(
    main_folder, n_processors, wall_path, centreline_path, save_filename,
    log_info
):
    logger.propagate = log_info
    # NOTE: modify this method to locate the case folders
    all_folders_to_process = [
        folder for folder in main_folder.glob("*") if folder.is_dir()]
    logger.info("Starting Unwrapping For %d Items"%len(all_folders_to_process))
    if n_processors == 1:
        [unwrap_vtk_and_save(folder) for folder in all_folders_to_process]
    else:
        with mp.Pool(n_processors) as p:
            resultsList = [
                p.apply_async(
                    unwrap_vtk_and_save, (folder, ),
                    kwds=dict(
                        centreline_path = centreline_path,
                        wall_path       = wall_path,
                        save_filename   = save_filename,
                        log_info        = log_info
                    )
                )
                for folder in all_folders_to_process
            ]
            for result in resultsList:
                result.get()

    logger.info("Completed Unwrapping For All")

if __name__ == "__main__":
    main()
