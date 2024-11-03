import torch
import numpy as np

from argparse import ArgumentParser
from arguments import (
    CompressionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussian_renderer import render
from utils.splats import to_full_cov, extract_rot_scale


def parse_arguments(simulated_args=[]):
    # Initialize the argument parser
    parser = ArgumentParser(description="Compression script parameters")
    
    # Add the same argument groups as in the script
    model = ModelParams(parser, sentinel=True)
    model.data_device = "cuda"
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    comp = CompressionParams(parser)
    
    # Combine simulated args with parser arguments
    args = get_combined_args(parser, simulated_args)
    return args, model, pipeline, op, comp



def determine_radius(scene, gaussians, resolution_factor=1.0):
    gaussian_scales = gaussians.get_scaling.amax(-1)
    
    # Calculate the mean position of all training cameras
    train_cameras = scene.getTrainCameras()
    camera_positions = torch.stack([cam.camera_center for cam in train_cameras])
    mean_camera_position = camera_positions.mean(dim=0)
    
    # Calculate distances of Gaussians from the mean camera position
    gaussian_positions = gaussians.get_xyz
    distances = torch.norm(gaussian_positions - mean_camera_position, dim=-1)
    
    # Calculate radius as a function of scale and distance
    radius = gaussian_scales / (distances + 1e-5) * resolution_factor
    
    return radius.mean().item()



def convert_centroids_to_rot_scale(centroids_history):
    rot_history = []
    scale_history = []

    # Decompose each centroid
    for centroids in centroids_history:
        centroids_tensor = torch.tensor(centroids, device="cuda") if not torch.is_tensor(centroids) else centroids
        rot, scale = extract_rot_scale(to_full_cov(centroids_tensor))
        rot_history.append(rot.cpu().numpy())
        scale_history.append(scale.cpu().numpy())

    return rot_history, scale_history



def select_best_camera(scene, dataset, pipe, scale=1.0):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    best_camera = None
    best_score = -np.inf
    
    # Iterate through each camera and evaluate the view
    for cam in scene.train_cameras[scale]:
        render_pkg = render(cam, scene.gaussians, pipe, background)
        rendered_image = render_pkg["render"].detach().cpu().numpy()
        
        # Compute a score based on image variance
        score = np.var(rendered_image)

        if score > best_score:
            best_score = score
            best_camera = cam

    return best_camera