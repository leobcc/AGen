from .networks import ImplicitNet, RenderingNet, GeometryEncodingNet
from .density import LaplaceDensity, AbsDensity
from .ray_sampler import ErrorBoundSampler
from .deformer import SMPLDeformer
from .smpl import SMPLServer

from .sampler import PointInSpace

from ..utils import utils

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import hydra
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces
class V2A(nn.Module):
    def __init__(self, opt, betas_path, gender, num_training_frames, implicit_network, geometry_encoding_network, rendering_network):
        super().__init__()

        # Foreground networks
        self.implicit_network = implicit_network
        self.geometry_encoding_network = geometry_encoding_network   # Video aspecific instance of the network, used for foreground
        self.rendering_network = rendering_network

        # Background networks
        self.bg_implicit_network = ImplicitNet(opt.bg_implicit_network)
        self.bg_rendering_network = RenderingNet(opt.bg_rendering_network)

        # Frame latent encoder
        self.frame_latent_encoder = nn.Embedding(num_training_frames, opt.bg_rendering_network.dim_frame_encoding)
        self.sampler = PointInSpace()

        betas = np.load(betas_path)
        self.use_smpl_deformer = opt.use_smpl_deformer
        self.gender = gender
        print("Using SMPL deformer: ", self.use_smpl_deformer)
        if self.use_smpl_deformer:
            self.deformer = SMPLDeformer(betas=betas, gender=self.gender)
        
        # pre-defined bounding sphere
        self.sdf_bounding_sphere = 3.0
        
        # threshold for the out-surface points
        self.threshold = 0.05
        
        self.density = LaplaceDensity(**opt.density)
        self.bg_density = AbsDensity()

        self.ray_sampler = ErrorBoundSampler(self.sdf_bounding_sphere, inverse_sphere_bg=True, **opt.ray_sampler)
        self.smpl_server = SMPLServer(gender=self.gender, betas=betas)

        # TODO: we need to initialize with smpl dict only on first training over the videos, later not
        if opt.smpl_init:
            smpl_model_state = torch.load(hydra.utils.to_absolute_path('VideoReconstructionModel/assets/smpl_init.pth'))
            self.implicit_network.load_state_dict(smpl_model_state["model_state_dict"])

        self.smpl_v_cano = self.smpl_server.verts_c
        self.smpl_f_cano = torch.tensor(self.smpl_server.smpl.faces.astype(np.int64), device=self.smpl_v_cano.device)

        self.mesh_v_cano = self.smpl_server.verts_c
        self.mesh_f_cano = torch.tensor(self.smpl_server.smpl.faces.astype(np.int64), device=self.smpl_v_cano.device)
        self.mesh_face_vertices = index_vertices_by_faces(self.mesh_v_cano, self.mesh_f_cano)

        # Initialize previous values for time consistency losses
        self.use_additional_losses = opt.loss.use_additional_losses
        if self.use_additional_losses and self.training:
            self.previous_canonical_surface_points = None
            self.previous_points = None
            self.previous_sdf_values = None
            self.previous_cond = None 
            self.previous_smpl_tfs = None
            self.previous_smpl_verts = None

    def sdf_func_with_smpl_deformer(self, x, frame_encoding_vector, uv_static, cond, smpl_tfs, smpl_verts):
        """ sdf_func_with_smpl_deformer method 
        Used to compute SDF values for input points using the SMPL deformer and the implicit network. 
        It handles the deforming of points, network inference, feature extraction, and handling of outlier points.
        """
        if hasattr(self, "deformer"):
            x_c, outlier_mask = self.deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
            #x_c_encoded = self.geometry_encoding_network(x_c, frame_encoding_vector)   # canonical points + frame encoding -> morphed canonical space
            if self.geometry_encoding_network is not None:
                x_c_morphed = self.geometry_encoding_network(x_c, frame_encoding_vector, uv_static)   # canonical points + frame encoding -> morphed canonical space
            else:
                x_c_morphed = x_c
            output = self.implicit_network(x_c_morphed, cond)[0]   # The geometry encoding is used only to improve sdf predictions, not to morph the canonical space
            morphing_distances = torch.mean(torch.norm(x_c_morphed - x_c, dim=1))
            #morphing_distances = self.implicit_network.morphing_distances(x_c, cond, frame_encoding_vector, uv_static)

            sdf = output[:, 0:1]
            feature = output[:, 1:]
            if not self.training:
                sdf[outlier_mask] = 4. # set a large SDF value for outlier points
        
        return sdf, x_c_morphed, feature, morphing_distances
    
    def check_off_in_surface_points_cano_mesh(self, x_cano, N_samples, threshold=0.05):
        """check_off_in_surface_points_cano_mesh method 
        Used to check whether points are off the surface or within the surface of a canonical mesh. 
        It calculates distances, signs, and signed distances to determine the position of points with respect to the mesh surface. 
        The method plays a role in identifying points that might be considered outliers or outside the reconstructed avatar's surface.
        """

        distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(x_cano.unsqueeze(0).contiguous(), self.mesh_face_vertices)

        distance = torch.sqrt(distance) # kaolin outputs squared distance
        sign = kaolin.ops.mesh.check_sign(self.mesh_v_cano, self.mesh_f_cano, x_cano.unsqueeze(0)).float()
        sign = 1 - 2 * sign   # -1 for off-surface, 1 for in-surface
        signed_distance = sign * distance
        batch_size = x_cano.shape[0] // N_samples  
        signed_distance = signed_distance.reshape(batch_size, N_samples, 1)   # The distances are reshaped to match the batch size and the number of samples

        minimum = torch.min(signed_distance, 1)[0]
        index_off_surface = (minimum > threshold).squeeze(1)
        index_in_surface = (minimum <= 0.).squeeze(1)
        return index_off_surface, index_in_surface   # Indexes of off-surface points and in-surface points
    
    def check_surface_cano_points(self, x_cano, sdf_values):
        sdf_values = sdf_values.reshape(x_cano.shape[0], x_cano.shape[1], 1)

        threshold = 1e-3 
        abs_sdf_values = torch.abs(sdf_values)
        mask_close_to_zero = abs_sdf_values < threshold

        return mask_close_to_zero

    def forward(self, input):
        # Parse model input, prepares the necessary input data and SMPL parameters for subsequent calculations
        torch.set_grad_enabled(True)
        intrinsics = input["intrinsics"]
        pose = input["pose"]
        uv = input["uv"]
        uv_static = input["uv"]

        frame_encoding_vector = input['frame_encoding_vector'].detach()   # The frame encoding tensor is extracted from the input dictionary

        scale = input['smpl_params'][:, 0]
        smpl_pose = input["smpl_pose"]
        smpl_shape = input["smpl_shape"]
        smpl_trans = input["smpl_trans"]
        smpl_output = self.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)   #  invokes the SMPL model to obtain the transformations for pose and shape changes

        smpl_tfs = smpl_output['smpl_tfs']

        cond = {'smpl': smpl_pose[:, 3:]/np.pi}
        if self.training:
            if input['current_epoch'] < 20 or input['current_epoch'] % 20 == 0:   # set the pose to zero for the first 20 epochs
                cond = {'smpl': smpl_pose[:, 3:] * 0.}  

        ray_dirs, cam_loc = utils.get_camera_params(uv, pose, intrinsics)   # get the ray directions and camera location

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)   # reshape to match the batch size and the number of pixels
        ray_dirs = ray_dirs.reshape(-1, 3)    # reshape to match the batch size and the number of pixels

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, frame_encoding_vector, uv_static, cond, smpl_tfs, eval_mode=True, smpl_verts=smpl_output['smpl_verts'])   # get the z values for each pixel  

        z_vals, z_vals_bg = z_vals   # unpack the z values for the foreground and the background
        z_max = z_vals[:,-1]   # get the maximum z value
        z_vals = z_vals[:,:-1]   # get the z values for the foreground

        N_samples = z_vals.shape[1]   # get the number of samples

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)   # 3D points along the rays are calculated by adding z_vals scaled by ray directions to the camera location. The result is stored in the points tensor of shape (batch_size * num_pixels, N_samples, 3)
        points_flat = points.reshape(-1, 3)   # The points tensor is reshaped into a flattened tensor points_flat of shape (batch_size * num_pixels * N_samples, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)   # The dirs tensor is created by repeating ray_dirs for each sample along the rays. The resulting tensor has shape (batch_size * num_pixels, N_samples, 3)
        sdf_output, canonical_points, feature_vectors, morphing_distances = self.sdf_func_with_smpl_deformer(points_flat, frame_encoding_vector, uv, cond, smpl_tfs, smpl_output['smpl_verts'])   # The sdf_func_with_smpl_deformer method is called to compute the signed distance functions (SDF) for the points
        
        sdf_output = sdf_output.unsqueeze(1)   # The sdf_output tensor is reshaped by unsqueezing along the first dimension

        if self.training:
            index_off_surface, index_in_surface = self.check_off_in_surface_points_cano_mesh(canonical_points, N_samples, threshold=self.threshold)
            canonical_points = canonical_points.reshape(num_pixels, N_samples, 3)   

            surface_cano_points_idx = self.check_surface_cano_points(canonical_points, sdf_output)

            canonical_points = canonical_points.reshape(-1, 3)   # The canonical points tensor flattened to shape (-1, 3) 
            surface_cano_points_idx = surface_cano_points_idx.reshape(-1, 1)

            # sample canonical SMPL surface pnts for the eikonal loss
            smpl_verts_c = self.smpl_server.verts_c.repeat(batch_size, 1,1)   # The canonical SMPL surface vertices are repeated across the batch dimension
            
            indices = torch.randperm(smpl_verts_c.shape[1])[:num_pixels].cuda()   # Random indices are generated to select a subset of vertices for sampling. The number of selected vertices is num_pixels
            verts_c = torch.index_select(smpl_verts_c, 1, indices)   # The selected vertices are gathered from smpl_verts_c, resulting in the tensor verts_c.
            sample = self.sampler.get_points(verts_c, global_ratio=0.)   # The get_points method of the sampler class is called to sample points around the canonical SMPL surface points. The global_ratio is set to 0.0, indicating local sampling

            sample.requires_grad_()   # The sampled points are marked as requiring gradients
            #sample_squeezed = sample.squeeze(0)
            #sample_encoded = self.geometry_encoding_network(sample_squeezed, frame_encoding_vector)   # The sampled points are passed through the geometry encoding network to obtain the conditioned sampled points 
            
            #sample = sample.squeeze(0)
            #sample = self.geometry_encoding_network(sample, frame_encoding_vector)
            #sample = sample.unsqueeze(0)
            local_pred = self.implicit_network(sample, cond)[..., 0:1]   # The sampled points (sample) are passed through the implicit network along with the conditioning (cond). The local prediction (SDF) for each sampled point is extracted using [..., 0:1]
            grad_theta = gradient(sample, local_pred)   # compute gradients with respect to the sampled points and their local predictions (local_pred).

            differentiable_points = canonical_points   # The differentiable_points tensor is assigned the value of canonical_points

        else:
            differentiable_points = canonical_points.reshape(num_pixels, N_samples, 3).reshape(-1, 3)
            grad_theta = None

        sdf_output = sdf_output.reshape(num_pixels, N_samples, 1).reshape(-1, 1)   # flattened to shape (num_pixels * N_samples, )
        z_vals = z_vals
        view = -dirs.reshape(-1, 3)   # The view vector is calculated as the negation of the reshaped dirs, giving the view directions for points along the rays.

        if differentiable_points.shape[0] > 0:   # If there are differentiable points (indicating that gradient information is available)
            fg_rgb_flat, others = self.get_rbg_value(points_flat, differentiable_points, view,
                                                     cond, frame_encoding_vector, uv_static, smpl_tfs, feature_vectors=feature_vectors, is_training=self.training)   # The returned values include fg_rgb_flat (foreground RGB values) and others (other calculated values, including normals)                 
            normal_values = others['normals']   # The normal values are extracted from the others dictionary

        if 'image_id' in input.keys():
            frame_latent_code = self.frame_latent_encoder(input['image_id'])
        else:
            frame_latent_code = self.frame_latent_encoder(input['idx'])

        fg_rgb = fg_rgb_flat.reshape(-1, N_samples, 3)
        normal_values = normal_values.reshape(-1, N_samples, 3)
        weights, bg_transmittance = self.volume_rendering(z_vals, z_max, sdf_output)

        fg_rgb_values = torch.sum(weights.unsqueeze(-1) * fg_rgb, 1)

        # Background rendering
        if input['idx'] is not None:
            N_bg_samples = z_vals_bg.shape[1]
            z_vals_bg = torch.flip(z_vals_bg, dims=[-1, ])  # 1--->0

            bg_dirs = ray_dirs.unsqueeze(1).repeat(1,N_bg_samples,1)
            bg_locs = cam_loc.unsqueeze(1).repeat(1,N_bg_samples,1)

            bg_points = self.depth2pts_outside(bg_locs, bg_dirs, z_vals_bg)  # [..., N_samples, 4]
            bg_points_flat = bg_points.reshape(-1, 4)
            bg_dirs_flat = bg_dirs.reshape(-1, 3)
            bg_output = self.bg_implicit_network(bg_points_flat, {'frame': frame_latent_code})[0]
            bg_sdf = bg_output[:, :1]
            bg_feature_vectors = bg_output[:, 1:]
            
            bg_rendering_output = self.bg_rendering_network(None, None, bg_dirs_flat, None, bg_feature_vectors, frame_latent_code)
            if bg_rendering_output.shape[-1] == 4:
                bg_rgb_flat = bg_rendering_output[..., :-1]
                shadow_r = bg_rendering_output[..., -1]
                bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)
                shadow_r = shadow_r.reshape(-1, N_bg_samples, 1)
                bg_rgb = (1 - shadow_r) * bg_rgb
            else:
                bg_rgb_flat = bg_rendering_output
                bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)
            bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sdf)
            bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, 1)
        else:
            bg_rgb_values = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)

        # Composite foreground and background
        bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
        rgb_values = fg_rgb_values + bg_rgb_values

        normal_values = torch.sum(weights.unsqueeze(-1) * normal_values, 1)

        if self.training:
            if self.use_additional_losses:
                if self.previous_canonical_surface_points == None:
                    self.previous_canonical_surface_points = canonical_points[surface_cano_points_idx.squeeze()].detach()
                    self.previous_points = points_flat.detach()
                    self.previous_sdf_values = sdf_output.detach()  
                    self.previous_cond = cond_copy = {key: value.clone().detach() for key, value in cond.items()}
                    self.previous_smpl_tfs = smpl_tfs.detach()
                    self.previous_smpl_verts = smpl_output['smpl_verts'].detach()
                sdf_values_on_previous_points, _, _, _ = self.sdf_func_with_smpl_deformer(self.previous_points, self.previous_cond, self.previous_smpl_tfs, self.previous_smpl_verts)   # sdf value on previous points, in order too compute the sdf soft time cons loss

                output = {
                    'points': points,

                    'previous_canonical_surface_points': self.previous_canonical_surface_points,
                    'canonical_surface_points': canonical_points[surface_cano_points_idx.squeeze()],
                    'previous_sdf_values': self.previous_sdf_values,
                    'sdf_values_on_previous_points': sdf_values_on_previous_points,
                    'previous_cond': self.previous_cond,
                    'previous_smpl_tfs': self.previous_smpl_tfs,
                    'previous_smpl_verts': self.previous_smpl_verts,

                    'rgb_values': rgb_values,
                    'normal_values': normal_values,
                    'index_outside': input['index_outside'],
                    'index_off_surface': index_off_surface,
                    'index_in_surface': index_in_surface,
                    'acc_map': torch.sum(weights, -1),
                    'sdf_output': sdf_output,
                    'grad_theta': grad_theta,
                    'epoch': input['current_epoch'],
                }
                self.previous_canonical_surface_points = canonical_points[surface_cano_points_idx.squeeze()].detach()
                self.previous_points = points_flat.detach()
                self.previous_sdf_values = sdf_output.detach()
                self.previous_cond = cond_copy = {key: value.clone().detach() for key, value in cond.items()}
                self.previous_smpl_tfs = smpl_tfs.detach()
                self.previous_smpl_verts = smpl_output['smpl_verts'].detach()
            else:
                output = {
                    'points': points,
                    'rgb_values': rgb_values,
                    'normal_values': normal_values,
                    'index_outside': input['index_outside'],
                    'index_off_surface': index_off_surface,
                    'index_in_surface': index_in_surface,
                    'acc_map': torch.sum(weights, -1),
                    'morphing_distances': morphing_distances,
                    'sdf_output': sdf_output,
                    'grad_theta': grad_theta,
                    'epoch': input['current_epoch'],
                }
        else:
            fg_output_rgb = fg_rgb_values + bg_transmittance.unsqueeze(-1) * torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
            output = {
                'acc_map': torch.sum(weights, -1),
                'rgb_values': rgb_values,
                'fg_rgb_values': fg_output_rgb,
                'normal_values': normal_values,
                'sdf_output': sdf_output,
            }
        return output

    def get_rbg_value(self, x, points, view_dirs, cond, frame_encoding_vector, uv_static, tfs, feature_vectors, is_training=True):
        pnts_c = points
        others = {}

        _, gradients, feature_vectors = self.forward_gradient(x, pnts_c, cond, frame_encoding_vector, uv_static, tfs, create_graph=is_training, retain_graph=is_training)
        # ensure the gradient is normalized
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        fg_rendering_output = self.rendering_network(pnts_c, normals, view_dirs, cond['smpl'],
                                                     feature_vectors)
        
        rgb_vals = fg_rendering_output[:, :3]
        others['normals'] = normals
        return rgb_vals, others

    def forward_gradient(self, x, pnts_c, cond, frame_encoding_vector, uv_static, tfs, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self.deformer.forward_skinning(pnts_c.unsqueeze(0), None, tfs).squeeze(0)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        #pnts_c_encoded = self.geometry_encoding_network(pnts_c, frame_encoding_vector)
        output = self.implicit_network(pnts_c, cond)[0]
        sdf = output[:, :1]
        
        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]

        return grads.reshape(grads.shape[0], -1), torch.nn.functional.normalize(torch.einsum('bi,bij->bj', gradients, grads_inv), dim=1), feature

    def volume_rendering(self, z_vals, z_max, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1]) # (batch_size * num_pixels) x N_samples

        # included also the dist from the sphere intersection
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[:, -1:]], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy], dim=-1)  # add 0 for transperancy 1 at t_0
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

    def bg_volume_rendering(self, z_vals_bg, bg_sdf):
        bg_density_flat = self.bg_density(bg_sdf)
        bg_density = bg_density_flat.reshape(-1, z_vals_bg.shape[1]) # (batch_size * num_pixels) x N_samples

        bg_dists = z_vals_bg[:, :-1] - z_vals_bg[:, 1:]
        bg_dists = torch.cat([bg_dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(bg_dists.shape[0], 1)], -1)

        # LOG SPACE
        bg_free_energy = bg_dists * bg_density
        bg_shifted_free_energy = torch.cat([torch.zeros(bg_dists.shape[0], 1).cuda(), bg_free_energy[:, :-1]], dim=-1)  # shift one step
        bg_alpha = 1 - torch.exp(-bg_free_energy)  # probability of it is not empty here
        bg_transmittance = torch.exp(-torch.cumsum(bg_shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        bg_weights = bg_alpha * bg_transmittance # probability of the ray hits something here

        return bg_weights
    
    def depth2pts_outside(self, ray_o, ray_d, depth):

        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - self.sdf_bounding_sphere ** 2)
        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm / self.sdf_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts

def gradient(inputs, outputs):

    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, :, -3:]
    return points_grad