The following scripts process data for a folder containing many cases.
The folder contains the following structure:


```
MainFolder
    Case_1
        boundary_1
            boundary_file.vtk
        boundary_2
            boundary_file.vtk
        ...
        boundary_n
            boundary_file.vtk
        internal_field.vtk
    Case_2
        boundary_1
            boundary_file.vtk
        boundary_2
            boundary_file.vtk
        ...
        boundary_n
            boundary_file.vtk
        internal_field.vtk
    ...
    Case_n
        boundary_1
            boundary_file.vtk
        boundary_2
            boundary_file.vtk
        ...
        internal_field.vtk

```

1. extract_centrelines_from_vtk.py
2. create_structured_grid_over_cylinder.py
3. unwarp_vtks.py
4. interpolate_cfd_to_structured.py
