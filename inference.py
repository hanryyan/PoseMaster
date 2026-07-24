import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import shutil
import trimesh
import numpy as np
import PIL.Image
from tqdm import tqdm
from posemaster.pipeline import PoseMasterPipeline
from posemaster.postprocessors import FloaterRemover, DegenerateFaceRemover
from posemaster.data.character_dataset_direction import load_skeleton

def save_ply_points(
        filename,
        points):
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % len(points))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for joint in points:
                f.write('%f %f %f\n' % (joint[0], joint[1], joint[2]))
              
def infer_pose(data_json, save_dir, config_path, ckpt_path):
    # initial
    PIPELINE = PoseMasterPipeline
    
    pipeline = PIPELINE.from_pretrained(
        config_path=config_path,
        dtype=torch.float16,
        device=torch.device("cuda"),
        ckpt_path=ckpt_path,
    )
    
    # split data
    print(f"save_dir: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    data = json.load(open(data_json))
    total_files = len(data['image'])

    for idx in range(total_files):
        image_file = data["image"][idx]
        uid = image_file.split("/")[-1].split(".")[0] + "_" + str(idx)
        print(uid)
        if not os.path.exists(image_file):
            print(f"{image_file} not exist")
            continue
        skeleton = data["skeleton"][idx]
        bone_points = load_skeleton(skeleton, num_bone_points=256, normalize_bone=True)['bone_points']

        result = pipeline(
            image=image_file,
            skeleton=bone_points,
            num_inference_steps=50,
            octree_depth=9,
            guidance_scale=4.5,
            seed=1234,
        )
        mesh = result['mesh']

        # save mesh
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh.export(os.path.join(save_dir, '%s.glb' % (uid)))

        # save skeleton
        save_ply_points(os.path.join(save_dir, '%s.ply' % uid), bone_points)

        # save image
        processed_image = result['image']
        processed_image.save(os.path.join(save_dir, '%s_processed.png' % uid))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on images using specified GPU.')
    parser.add_argument('--data_json', type=str, default="val_data/val_data.json", help='Directory of input images')
    parser.add_argument('--save_dir', type=str, default="val_data/infrence_results", help='Directory to save output files')
    parser.add_argument('--config_path', type=str, default="configs/shape-diffusion/shape-diff-hunyuan3d-bone-query-256-lr1e5-freeze-skeleton.yaml", help='Config of model')
    parser.add_argument('--ckpt_path', type=str, default="ckpts/posemaster_dit.ckpt", help='Weight of model')
    args = parser.parse_args()

    
    infer_pose(args.data_json, args.save_dir, args.config_path, args.ckpt_path)


