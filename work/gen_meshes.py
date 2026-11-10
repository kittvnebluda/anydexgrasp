"""Hand meshes and point clouds per grasp type and width, keeping the shipped width_12Dangle_6Dangle.json."""

import os

import adg_paths

os.chdir(adg_paths.ROOT)
import recover_inspire_hand_to_stl as r

base = "./generate_mesh_and_pointcloud/inspire_urdf/"
angles = r.read_excel_6Dangle_to_12Dangle(base + "inspire_hand_routine_to_angle-use.xlsx", base + "driver_routine_to_angle.xls")
r.get_meshes(angles, base + "urdf-five3/meshes/", base + "meshes/", None, base + "urdf-five3/robots/urdf-five3.urdf", if_source=False, vis=False)
