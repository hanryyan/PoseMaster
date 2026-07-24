import numpy as np
import os
import tqdm
import glob
import trimesh
# import trimesh


if __name__ == "__main__":
    task_dir = "output"
    methods = ["gt_mesh", "Trellis"] # 
    for method in methods:
        if method == "gt_mesh":
            meshes = glob.glob(os.path.join(task_dir, "gt_mesh", "*"))
        else:
            meshes = glob.glob(os.path.join(task_dir, method, "*.glb"))

        for mesh in meshes:
            uid = mesh.split("/")[-1].split(".")[0]
            result_base_dir = os.path.join(task_dir, "render", method, mesh.split("/")[-1].split(".")[0])
            print(f"render file: {mesh}")
            os.makedirs(result_base_dir, exist_ok=True)
            os.system(f"blenderproc run render/blenderproc_render_singleton.py --model_file {mesh} --render_config_file render/render_config_apose.json --result_base_dir {result_base_dir}")
                # exit()