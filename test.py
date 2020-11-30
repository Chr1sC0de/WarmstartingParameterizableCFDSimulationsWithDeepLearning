# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Creating Structured Internal Fields from VTK
# To create the structured internal field we can extract the centreline of the boundary and slice along it at equal intervals to obtain equally spaced contours. We can then parameterize the boundary of the domain and interpolate a set of structured datapoints using [transfinite interpolation](https://en.wikipedia.org/wiki/Transfinite_interpolation#:~:text=In%20numerical%20analysis%2C%20transfinite%20interpolation,field%20of%20finite%20element%20method.). By controlling the density of the interpolation we can control the overall accuracy of the warm-start prediction.
# %% [markdown]
# After completing a CFD simulation in OpenFoam we can make processing the case simpler by converting the cases into VTK format using the command 'foamToVTK'. The resulting folder will have the following structure:
#
# ![](./images/foam_vtk_example.png)
# %% [markdown]
# We can now easily view and interact with the resulting CFD data in python using pyvista

# %%
import pathlib as pt
import pyvista as pv
wall_path = pt.Path("./test_data/00000/WALL/WALL_400.vtk")
wall_data = pv.read(wall_path)

# %% [markdown]
# # Extracting the Centreline
# We can extract the centreline using CGALMethods

# %%
# unwrap the artery
import pymethods as pma
import CGALMethods as cm
import pathlib as pt
case_folder = pt.Path("./test_data/00000")
surface_mesh_path = case_folder/"WALL/WALL_400.vtk"
inlet_mesh_path = case_folder/"INLET/INLET_400.vtk"
# load the mesh as a CGAL surface mesh, we can use methods within the cgal library to extract the centreline of the mesh
cgal_mesh = cm.SurfaceMesh(surface_mesh_path.as_posix())
inlet_mesh = cm.SurfaceMesh(inlet_mesh_path.as_posix())

# %% [markdown]
# For some CGAL mesh we can view the property names with the property names method

# %%
inlet_mesh.property_names()

# %% [markdown]
# We can extract some field from the property mesh with the "get_property" method

# %%
inlet_velocity_field = inlet_mesh.get_property("v:U")
print(inlet_velocity_field.shape)

# %% [markdown]
# And we can extract the points of the mesh by calling the "points" method

# %%
inlet_points = inlet_mesh.points()
print(inlet_points.shape)


# %%
# approximate the centre of the inlet
inlet_centroid = inlet_points.mean(0)


# %%
# now we can extract the centreline of the mesh
centreline_raw = pma.algorithms.unwrapping.get_centerline_from_cylindrical_mesh(cgal_mesh,inlet_origin=inlet_centroid)


# %%
import matplotlib.pyplot as plt
import numpy as np
# make the plot shown an interactive widget
# convert the centreline to a Curve object
centreline = pma.arrays.Curve(centreline_raw)
# centreline.plot3d(".", label="original")
# now divide the centreline into 300 equally spaced points
centreline = centreline(np.linspace(0,1, 300))
# centreline.plot3d(".", label="interpolated")
# plt.legend()
# plt.show()

# %% [markdown]
# # Unwrapping the Mesh Using The Centreline
# Once we have the centreline using CGALMethods we can process the mesh in pyvista

# %%
point_grid, field_grid = pma.algorithms.unwrapping.unwrap_cylinder_vtk_from_centerline(centreline, wall_data)


# %%



