# powershell script for processing all the data within a folder using the python
# scripts

python .\extract_centrelines_from_vtk.py --main_folder I:\VTK
python .\unwrap_vtks.py --main_folder I:\VTK
python .\create_structured_grid_over_cylinder.py --main_folder I:\VTK
python .\interpolate_cfd_to_structured.py --main_folder I:\VTK