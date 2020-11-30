import pathlib as pt
import pymethods as pma
import multiprocessing as mp
import CGALMethods as cm
import numpy as np
import click
from scipy.spatial import cKDTree
import pyvista as pv
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger()

def string_to_list_ints(data):
    try:
        numbers = data.strip("()")
        numbers = [int(number) for number in data.split(",")]
        return numbers
    except:
        return data

def interpolate_cfd_onto_structured_mesh(
    folder            : pt.Path,
    original_mesh_path: str = "FOAM_400.vtk",
    structured_path   : str = "structured_volume.npy",
    data_save_filename: str = "structured_interpolated_data.npz",
    vtk_save_filename : str = "vtk_interpolated_data.vtk",
    test_neighbours   : int or str = range(2, 11),
    log_info          : bool = True
):
    logger.propagate = log_info
    logger.info("Processing %s"%folder.name)

    original_mesh         = pv.read(folder/original_mesh_path)
    structured_mesh       = np.load(folder/structured_path)

    list_structured_mesh_internal = structured_mesh.reshape(3, -1, order="F").T

    U_orig = original_mesh.point_arrays["U"]
    p_orig = original_mesh.point_arrays["p"]

    original_mesh_kdtree   = cKDTree(original_mesh.points)
    structured_mesh_kdtree = cKDTree(list_structured_mesh_internal)

    l = {}

    for neighbours in test_neighbours:

        distances, indices = original_mesh_kdtree.query(
            list_structured_mesh_internal, neighbours)
        # now apply inverse distance weighting interpolation
        U_interpolated = ((U_orig[indices]/distances[:, :, None]).sum(1))/(1/distances).sum(1)[:,None]
        p_interpolated = ((p_orig[indices]/distances).sum(1))/(1/distances).sum(1)
        # reinterpolate the structured mesh onto the original mesh
        distances, indices = structured_mesh_kdtree.query(original_mesh.points, neighbours)
        U_back_interpolated = ((U_interpolated[indices]/distances[:, :, None]).sum(1))/(1/distances).sum(1)[:,None]
        p_back_interpolated = ((p_interpolated[indices]/distances).sum(1))/(1/distances).sum(1)
        loss = (
            ((U_orig-U_back_interpolated)**2).mean() + ((p_orig-p_back_interpolated)**2).mean()
        )

        l[loss] = {"p": p_interpolated, "U": U_interpolated, "neighbours": neighbours}

    l_best = np.min(list(l.keys()))
    structured_mesh_shape =structured_mesh.shape
    p_best = l[l_best]["p"].T.reshape(*structured_mesh_shape[1:], order="F")[None, :, :, :]
    U_best = l[l_best]["U"].T.reshape(*structured_mesh_shape, order="F")
    n_best = l[l_best]["neighbours"]

    np.savez(
        folder/data_save_filename,
        points = structured_mesh,
        U      = U_best,
        p      = p_best
    )

    mesh = pv.StructuredGrid(*structured_mesh)

    mesh.point_arrays["p"] = l[l_best]["p"]
    mesh.point_arrays["U"] = l[l_best]["U"]

    mesh.save(folder/vtk_save_filename)

    logger.info("%s completed with %d neighbours"%(folder.name, n_best))

@click.command()
@click.option("--main_folder", type=pt.Path, default=".")
@click.option("--n_processors", default=1, type=int)
@click.option("--save_filename", default="structured_interpolated_data.npz", type=str)
@click.option("--vtk_save_filename", default="vtk_interpolated.vtk", type=str)
# @click.option("--test_neighbours", default=range(2, 11), type=string_to_list_ints)
@click.option("--test_neighbours", default=[4,], type=string_to_list_ints)
@click.option("--log_info", default=bool)
def main(
    main_folder, n_processors, save_filename, vtk_save_filename, test_neighbours, log_info
):
    logger.propagate = log_info
    # NOTE: moify this method to locate the case folders
    all_folders_to_process = [
        folder for folder in main_folder.glob("*") if folder.is_dir()]
    logger.info("Starting Interpolation For %d Items"%len(all_folders_to_process))
    if n_processors == 1:
        [
            interpolate_cfd_onto_structured_mesh(folder) for
            folder in all_folders_to_process
        ]
    else:
        with mp.Pool(n_processors) as p:
            resultsList = [
                p.apply_async(
                    interpolate_cfd_onto_structured_mesh, (folder, )
                )
                for folder in all_folders_to_process
            ]
            for result in resultsList:
                result.get()

    logger.info("Completed Interpolation For All")

if __name__ == "__main__":
    main()
