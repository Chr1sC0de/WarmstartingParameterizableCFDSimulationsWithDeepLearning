import pathlib as pt
import pymethods as pma
import multiprocessing as mp
import CGALMethods as cm
import numpy as np
import click
import logging

'''
    The following extracts centrelines of all cases within a folder
    containing multiple vtks of openfoam artery CFD cases.
'''

logger = logging.getLogger()

def extract_centreline_vtk(
    folder       : pt.Path,
    wall_path    : str = "WALL/WALL_400.vtk",
    inlet_path   : str = "INLET/INLET_400.vtk",
    save_filename: str = "centreline",
    log_info     : bool = True
):
    logger.propagate = log_info

    logger.info("Processing: %s"%folder.name)

    wall_path   = folder/wall_path
    inlet_path  = folder/inlet_path

    cgal_surface_mesh = cm.SurfaceMesh(wall_path.as_posix())
    inlet_mesh        = cm.SurfaceMesh(inlet_path.as_posix())

    inlet_centroid = inlet_mesh.points().mean(0)

    centreline_raw = pma.algorithms.unwrapping.get_centerline_from_cylindrical_mesh(
        cgal_surface_mesh, inlet_origin=inlet_centroid
    )

    np.save(folder/save_filename, centreline_raw)

    logger.info("Completed Processing Case: %s"%folder.name)


@click.command()
@click.option("--main_folder", type=pt.Path, default=".", help="main_folder for all the cases")
@click.option("--n_processors", default=8, type=int, help="the total number of parallel processes to performs")
@click.option("--wall_path", default="WALL/WALL_400.vtk", type=str, help="folder and filename of the wall")
@click.option("--inlet_path", default="INLET/INLET_400.vtk", type=str, help="the total number of parallel processes to performs")
@click.option("--save_filename", default="centreline", type=str, help="the total number of parallel processes to performs")
@click.option("--log_info", default="true", type=bool, help="the total number of parallel processes to performs")
def main(main_folder, n_processors, wall_path, inlet_path, save_filename, log_info):

    logger.propagate = log_info
    # NOTE: moify this method to locate the case folders
    all_folders_to_process = [
        folder for folder in main_folder.glob("*") if folder.is_dir()]

    logger.info("Starting Centreline Extraction For %d Items"%len(all_folders_to_process))

    with mp.Pool(n_processors) as p:
        resultsList = [
            p.apply_async(
                extract_centreline_vtk, (folder, ),
                kwds=dict(
                    wall_path     = wall_path,
                    inlet_path    = inlet_path,
                    save_filename = save_filename,
                    log_info      = log_info
                )
            )
            for folder in all_folders_to_process
        ]
        for result in resultsList:
            result.get()

    logger.info("Completed Centreline All")

if __name__ == "__main__":
    main()