# -*- coding: utf-8 -*-
"""
Tencent is pleased to support the open source community by making Tencent Hunyuan 3D Omni available.

Copyright (C) 2025 Tencent.  All rights reserved. The below software and/or models in this 
distribution may have been modified by Tencent ("Tencent Modifications"). All Tencent Modifications 
are Copyright (C) Tencent.

Tencent Hunyuan 3D Omni is licensed under the TENCENT HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT 
except for the third-party components listed below, which is licensed under different terms. 
Tencent Hunyuan 3D Omni does not impose any additional limitations beyond what is outlined in the 
respective licenses of these third-party components. Users must comply with all terms and conditions 
of original licenses of these third-party components and must ensure that the usage of the third party 
components adheres to all relevant laws and regulations. 

For avoidance of doubts, Tencent Hunyuan 3D Omni means training code, inference-enabling code, parameters, 
and/or weights of this Model, which are made publicly available by Tencent in accordance with TENCENT 
HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT.
"""

import os
import tempfile
import time
from typing import Union

import numpy as np
import pymeshlab
import requests
import torch
import trimesh

class Latent2MeshOutput(object):
    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None


def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path, process=False)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=True) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    # 检查加载的对象类型
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        # 如果是Scene，遍历所有的geometry并合并
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=True) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)
    return mesh


def export_mesh(
    input: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput],
    output: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput],
) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
    """ convert output mesh to the type of the input mesh"""
    if type(input) == type(output):
        return output

    if isinstance(input, Latent2MeshOutput):
        mesh = Latent2MeshOutput()
        if isinstance(output, trimesh.Trimesh):
            mesh.mesh_v = output.mesh_v
            mesh.mesh_f = output.mesh_f
        elif isinstance(output, pymeshlab.MeshSet):
            mesh.mesh_v = output.current_mesh().vertex_matrix()
            mesh.mesh_f = output.current_mesh().face_matrix()
        else:
            mesh = output
    elif isinstance(input, pymeshlab.MeshSet):
        if isinstance(output, Latent2MeshOutput):
            output = trimesh.Trimesh(vertices=output.mesh_v, faces=output.mesh_f)
        if isinstance(output, trimesh.Trimesh):
            mesh = trimesh2pymeshlab(output)
        else:
            mesh = output
    elif isinstance(input, (trimesh.Trimesh, str)):
        if isinstance(output, Latent2MeshOutput):
            output = trimesh.Trimesh(vertices=output.mesh_v, faces=output.mesh_f)
        if isinstance(output, pymeshlab.MeshSet):
            mesh = pymeshlab2trimesh(output)
        else:
            mesh = output
    else:
        raise Exception("input type not supported")
    return mesh


def import_pymeshlab_mesh(mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str]) -> pymeshlab.MeshSet:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.mesh_v, face_matrix=mesh.mesh_f)
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh


def import_trimesh_mesh(mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str]) -> trimesh.Trimesh:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        mesh = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
    if isinstance(mesh, pymeshlab.MeshSet):
        mesh = pymeshlab2trimesh(mesh)
    return mesh


class FaceReducer:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
        max_facenum: int = 10000
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        ms = import_pymeshlab_mesh(mesh)
        ms = reduce_face(ms, max_facenum=max_facenum)
        mesh = export_mesh(mesh, ms)
        return mesh


def mesh_normalize(mesh):
    """
    Normalize mesh vertices to sphere
    """
    scale_factor = 1.2
    vtx_pos = np.asarray(mesh.vertices)
    max_bb = (vtx_pos - 0).max(0)[0]
    min_bb = (vtx_pos - 0).min(0)[0]

    center = (max_bb + min_bb) / 2

    scale = torch.norm(torch.tensor(vtx_pos - center, dtype=torch.float32), dim=1).max() * 2.0

    vtx_pos = (vtx_pos - center) * (scale_factor / float(scale))
    mesh.vertices = vtx_pos

    return mesh


class FloaterRemover:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_pymeshlab_mesh(mesh)
        ms = remove_floater(ms)
        mesh = export_mesh(mesh, ms)
        return mesh


class DegenerateFaceRemover:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_pymeshlab_mesh(mesh)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=True) as temp_file:
            ms.save_current_mesh(temp_file.name)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_file.name)

        mesh = export_mesh(mesh, ms)
        return mesh