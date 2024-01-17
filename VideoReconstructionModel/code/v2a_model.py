from lib.model.v2a import V2A
from lib.model.body_model_params import BodyModelParams
from lib.model.deformer import SMPLDeformer
from lib.model.loss import Loss
from lib.utils.meshing import generate_mesh
from lib.model.deformer import skinning
from lib.utils import utils
from evaluation import compute_ssim, compute_psnr

import pytorch_lightning as pl
import torch.optim as optim
import cv2
import torch
import hydra
import os
import numpy as np
from kaolin.ops.mesh import index_vertices_by_faces
import trimesh

class V2AModel(pl.LightningModule):
    def __init__(self, opt, implicit_network: torch.nn.Module, geometry_encoding_network: torch.nn.Module, rendering_network: torch.nn.Module) -> None:
        super().__init__()

        self.opt = opt
        num_training_frames = opt.dataset.metainfo.end_frame - opt.dataset.metainfo.start_frame
        self.betas_path = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), 'data', opt.dataset.metainfo.data_dir, 'mean_shape.npy'))
        self.gender = opt.dataset.metainfo.gender
        self.model = V2A(opt.model, self.betas_path, self.gender, num_training_frames, implicit_network, geometry_encoding_network, rendering_network)
        self.start_frame = opt.dataset.metainfo.start_frame
        self.end_frame = opt.dataset.metainfo.end_frame
        self.training_modules = ["model"]
        self.N_validation_batches = opt.videos_dataset.valid.N_validation_batches

        self.training_indices = list(range(self.start_frame, self.end_frame))
        self.body_model_params = BodyModelParams(num_training_frames, model_type='smpl')
        self.load_body_model_params()
        optim_params = self.body_model_params.param_names
        for param_name in optim_params:
            self.body_model_params.set_requires_grad(param_name, requires_grad=True)
        self.training_modules += ['body_model_params']
        
        self.loss = Loss(opt.model.loss)

        self.average_val_met = {f'ssim': 0,
                                f'psnr': 0,
                                f'mask_ssim': 0,
                                f'mask_psnr': 0,
                                f'ssim_train': 0,
                                f'psnr_train': 0}

    def load_body_model_params(self):
        body_model_params = {param_name: [] for param_name in self.body_model_params.param_names}
        data_root = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), 'data', self.opt.dataset.metainfo.data_dir))
        data_root = hydra.utils.to_absolute_path(data_root)

        body_model_params['betas'] = torch.tensor(np.load(os.path.join(data_root, 'mean_shape.npy'))[None], dtype=torch.float32)
        body_model_params['global_orient'] = torch.tensor(np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices][:, :3], dtype=torch.float32)
        body_model_params['body_pose'] = torch.tensor(np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices] [:, 3:], dtype=torch.float32)
        body_model_params['transl'] = torch.tensor(np.load(os.path.join(data_root, 'normalize_trans.npy'))[self.training_indices], dtype=torch.float32)

        for param_name in body_model_params.keys():
            self.body_model_params.init_parameters(param_name, body_model_params[param_name], requires_grad=False) 

    def configure_optimizers(self):
        params = [{'params': self.model.parameters(), 'lr':self.opt.model.learning_rate}]
        params.append({'params': self.body_model_params.parameters(), 'lr':self.opt.model.learning_rate*0.1})
        self.optimizer = optim.Adam(params, lr=self.opt.model.learning_rate, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)
        return [self.optimizer], [self.scheduler]
    
    def forward(self, x):
        inputs, targets = x   # One batch (frame) is passed in x

        batch_idx = inputs["idx"]

        body_model_params = self.body_model_params(batch_idx)
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        inputs['current_epoch'] = self.current_epoch
        
        model_outputs = self.model(inputs)

        return model_outputs

    def training_step(self, batch):
        inputs, targets = batch

        batch_idx = inputs["idx"]

        body_model_params = self.body_model_params(batch_idx)
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        inputs['current_epoch'] = self.current_epoch
        model_outputs = self.model(inputs)

        loss_output = self.loss(model_outputs, targets)
        for k, v in loss_output.items():
            if k in ["loss"]:
                self.log(k, v.item(), prog_bar=True, on_step=True)
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)
        return loss_output["loss"]

    # TODO: This is possibly not necessary during training, but it might be necessary on the outer model for training
    def training_epoch_end(self, outputs) -> None:       
        # Canonical mesh update every 20 epochs
        if self.current_epoch != 0 and self.current_epoch % 20 == 0:
            cond = {'smpl': torch.zeros(1, 69).float().cuda()}
            data_root = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), 'data', self.opt.dataset.metainfo.data_dir))
            frame_encoding_vector = torch.tensor(np.load(os.path.join(data_root, 'feature_encoding_vectors.npy'))[0], dtype=torch.float32).cuda()
            mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=2)
            self.model.mesh_v_cano = torch.tensor(mesh_canonical.vertices[None], device = self.model.smpl_v_cano.device).float()
            self.model.mesh_f_cano = torch.tensor(mesh_canonical.faces.astype(np.int64), device=self.model.smpl_v_cano.device)
            self.model.mesh_face_vertices = index_vertices_by_faces(self.model.mesh_v_cano, self.model.mesh_f_cano)

        return super().training_epoch_end(outputs)

    # TODO: This goes with the generation of the canonical mesh on training_epoch_end, same reasoning, but it's also used in other places down here
    def query_oc(self, x, cond):
        
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.implicit_network(x, cond)[:,:,0].reshape(-1,1)
        return {'sdf':mnfld_pred}

    def query_wc(self, x):
        
        x = x.reshape(-1, 3)
        w = self.model.deformer.query_weights(x)
    
        return w

    def query_od(self, x, cond, smpl_tfs, smpl_verts):
        
        x = x.reshape(-1, 3)
        x_c, _ = self.model.deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
        #x_c_encoded = self.geometry_encoding_network(x_c, frame_encoding_vector)   # canonical points + frame encoding -> morphed canonical space
        output = self.model.implicit_network(x_c_encoded, cond, frame_encoding_vector)[0]
        sdf = output[:, 0:1]
        
        return {'sdf': sdf}

    def get_deformed_mesh_fast_mode(self, verts, smpl_tfs):
        verts = torch.tensor(verts).cuda().float()
        weights = self.model.deformer.query_weights(verts)
        verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def sampling_from_mesh(self, dataloader, split):
        '''
        This function is used to sample points from the mesh
        It is used for the refinement step before training on unseen data
        It does not make sense to use it if the networks are not pretrained
        '''
        sampled_points = []
        for idx, batch in enumerate(dataloader):
            '''
            output = {}
            inputs, targets = batch
            self.model.eval()

            body_model_params = self.body_model_params(inputs["idx"])
            inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1).cuda()
            inputs['smpl_shape'] = body_model_params['betas'].cuda()
            inputs['smpl_trans'] = body_model_params['transl'].cuda()

            cond = {'smpl': inputs["smpl_pose"][:, 3:].cuda()/np.pi}
            mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=split.num_sample, res_up=3)
            '''
            inputs, targets = batch
            scale, smpl_trans, smpl_pose, smpl_shape = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=1)

            body_model_params = self.body_model_params(inputs['idx'])
            smpl_shape = body_model_params['betas'] if body_model_params['betas'].dim() == 2 else body_model_params['betas'].unsqueeze(0).cuda()
            smpl_trans = body_model_params['transl'].cuda()
            smpl_pose = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1).cuda()

            smpl_outputs = self.model.smpl_server(scale.cuda(), smpl_trans.cuda(), smpl_pose.cuda(), smpl_shape.cuda())
            smpl_tfs = smpl_outputs['smpl_tfs']
            cond = {'smpl': smpl_pose[:, 3:]/np.pi}

            mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=split.num_sample, res_up=4)

            self.model.deformer = SMPLDeformer(betas=np.load(self.betas_path), gender=self.gender, K=7)
            verts_deformed = self.get_deformed_mesh_fast_mode(mesh_canonical.vertices, smpl_tfs)
            print("verts_deformed:", verts_deformed.shape)
            print("verts_deformed range:", verts_deformed.min(), verts_deformed.max())
            mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)

            print("img.shape", inputs['frame_encoding_vector'].shape)
            print("mesh_deformed.vertices.shape:", mesh_deformed.vertices.shape)
            points = np.array(mesh_deformed.vertices[np.random.choice(mesh_deformed.vertices.shape[0], size=split.num_sample, replace=False)])
            print("points.shape", points.shape)
            cam_loc = utils.get_camera_center(inputs['pose'])
            print("cam_loc.shape:", cam_loc.shape)
            rays = points - cam_loc.numpy()
            print("rays.shape:", rays.shape)
            z_min = np.min(rays[:, 2])
            samples_normalized = rays[:, :2]*z_min/(rays[:, 2])[:, np.newaxis]   # x' = x*z_min/z, y' = y*z_min/z
            print("samples.shape", samples_normalized.shape)
            print("samples", samples_normalized)
            print("range_y", np.min(samples_normalized[:, 0]), np.max(samples_normalized[:, 0]))
            print("range_x", np.min(samples_normalized[:, 1]), np.max(samples_normalized[:, 1]))

            img_height, img_width = 1080, 1920
            min_y, min_x = np.min(samples_normalized, axis=0)
            max_y, max_x = np.max(samples_normalized, axis=0)

            #amples = ((samples_normalized - np.array([min_x, min_y])) / np.array([max_y - min_y, max_x - min_x])) * np.array([img_height, img_width])   # from [min_x, max_x] to [0, img_width] and from [min_y, max_y] to [0, img_height]
            samples = ((samples_normalized + 3) / 6) * np.array([img_height, img_width])   # from [-3, 3] to [0, img_width] and from [-3, 3] to [0, img_height]
            sampled_points.append(samples)

            print("range_y", np.min(samples[:, 0]), np.max(samples[:, 0]))
            print("range_x", np.min(samples[:, 1]), np.max(samples[:, 1]))
            if (np.min(samples[:, 0]) < 0) or (np.max(samples[:, 0]) > img_height) or (np.min(samples[:, 1]) < 0) or (np.max(samples[:, 1]) > img_width):
                print("ERROR: samples out of range---------------------------------------------------")
                print("idx:", idx)
                break
            print("type:", type(sampled_points[idx]))
            print("finished sampling for batch", idx)

        return sampled_points

    # TODO: The validation step is very memory intensive, not allowing to scale during training. It should be optimized to return only the necessary outputs for outer training
    # Validation -----------------------------------------------------------------------------
    '''This validation is performed on one frame (batch) of the video being processed'''
    def validation_step(self, batch, *args, **kwargs):
        output = {}
        inputs, targets = batch
        inputs['current_epoch'] = self.current_epoch
        self.model.eval()

        body_model_params = self.body_model_params(inputs['image_id'])
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        cond = {'smpl': inputs["smpl_pose"][:, 3:]/np.pi}
        mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=3)
        
        mesh_canonical = trimesh.Trimesh(mesh_canonical.vertices, mesh_canonical.faces)
        
        output.update({
            'canonical_mesh':mesh_canonical
        })

        split = utils.split_input(inputs, targets["total_pixels"][0], n_pixels=min(targets['pixel_per_batch'], targets["img_size"][0] * targets["img_size"][1]))

        res = []
        for s in split:

            out = self.model(s)

            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v

            res.append({
                'rgb_values': out['rgb_values'].detach(),
                'normal_values': out['normal_values'].detach(),
                'fg_rgb_values': out['fg_rgb_values'].detach(),
            })
        batch_size = targets['rgb'].shape[0]

        model_outputs = utils.merge_output(res, targets["total_pixels"][0], batch_size)

        output.update({
            "rgb_values": model_outputs["rgb_values"].detach().clone(),
            "normal_values": model_outputs["normal_values"].detach().clone(),
            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
            **targets,
        })
            
        return output

    def validation_step_end(self, batch_parts):
        
        return batch_parts
        
    # TODO: writing the files here should not be necessary and it's only a waste of memory
    def validation_epoch_end(self, outputs):
        # Writing output files -------------------------------------------------------------------
        img_size = outputs[0]["img_size"]

        rgb_pred = torch.cat([output["rgb_values"] for output in outputs], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([output["fg_rgb_values"] for output in outputs], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([output["normal_values"] for output in outputs], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        rgb_gt = torch.cat([output["rgb"] for output in outputs], dim=1).squeeze(0)
        rgb_gt = rgb_gt.reshape(*img_size, -1)
        if 'normal' in outputs[0].keys():
            normal_gt = torch.cat([output["normal"] for output in outputs], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb_comparison = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        rgb_comparison = (rgb_comparison * 255).astype(np.uint8)   # Return to range [0, 255]

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        if self.opt.dataset.metainfo.type == 'test-non-pretrained':
            video_directory_path = os.path.join('Video/', self.opt.dataset.metainfo.data_dir, 'non-pretrained/')
        else:
            video_directory_path = os.path.join('Video/', self.opt.dataset.metainfo.data_dir)
        os.makedirs(os.path.join(video_directory_path, 'rendering'), exist_ok=True)
        os.makedirs(os.path.join(video_directory_path, 'normal'), exist_ok=True)
        os.makedirs(os.path.join(video_directory_path, 'fg_rendering'), exist_ok=True)

        canonical_mesh = outputs[0]['canonical_mesh']
        canonical_mesh.export(os.path.join(video_directory_path, f"rendering/{self.current_epoch}.ply"))

        cv2.imwrite(os.path.join(video_directory_path, f"rendering/{self.current_epoch}.png"), rgb_comparison[:, :, ::-1])
        cv2.imwrite(os.path.join(video_directory_path, f"normal/{self.current_epoch}.png"), normal[:, :, ::-1])
        cv2.imwrite(os.path.join(video_directory_path, f"fg_rendering/{self.current_epoch}.png"), fg_rgb[:, :, ::-1])

        # Computing evaluation metrics ----------------------------------------------------------- 
        #rgb_pred_np = rgb_pred.permute(2, 0, 1).cpu().numpy()  # Convert from (H, W, C) to (C, H, W)
        #rgb_gt_np = rgb_gt.permute(2, 0, 1).cpu().numpy()
        print("rgb shapes:", rgb_pred.shape, rgb_gt.shape)
        rgb_pred_np_greysc = cv2.cvtColor(rgb_pred.cpu().numpy(), cv2.COLOR_BGR2GRAY)   
        rgb_gt_np_greysc = cv2.cvtColor(rgb_gt.cpu().numpy(), cv2.COLOR_BGR2GRAY)
        rgb_pred_np = rgb_pred.cpu().numpy() 
        rgb_gt_np = rgb_gt.cpu().numpy()
        print("images shapes:", rgb_pred_np.shape, rgb_gt_np.shape)

        # Compute metrics
        validation_metrics = {
            f'ssim_train': compute_ssim(rgb_gt_np_greysc, rgb_pred_np_greysc),   # Normalized rgb values are passed
            f'psnr_train': compute_psnr(rgb_gt_np, rgb_pred_np),    # Normalized rgb values are passed
        }

        # Log the metrics values
        for k, v in validation_metrics.items():
            self.average_val_met[k] = self.average_val_met[k] + v
            print(f'{k}: {v}')

        return validation_metrics
    
    def test_step(self, batch, *args, **kwargs):
        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_splits = (total_pixels + pixel_per_batch -
                       1) // pixel_per_batch
        results = []

        scale, smpl_trans, smpl_pose, smpl_shape = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=1)

        body_model_params = self.body_model_params(inputs['idx'])
        smpl_shape = body_model_params['betas'] if body_model_params['betas'].dim() == 2 else body_model_params['betas'].unsqueeze(0)
        smpl_trans = body_model_params['transl']
        smpl_pose = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)

        smpl_outputs = self.model.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_outputs['smpl_tfs']
        cond = {'smpl': smpl_pose[:, 3:]/np.pi}

        mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=4)
        self.model.deformer = SMPLDeformer(betas=np.load(self.betas_path), gender=self.gender, K=7)
        verts_deformed = self.get_deformed_mesh_fast_mode(mesh_canonical.vertices, smpl_tfs)
        mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)

        os.makedirs("test_mask", exist_ok=True)
        os.makedirs("test_rendering", exist_ok=True)
        os.makedirs("test_fg_rendering", exist_ok=True)
        os.makedirs("test_normal", exist_ok=True)
        os.makedirs("test_mesh", exist_ok=True)
        
        mesh_canonical.export(f"test_mesh/{int(idx.cpu().numpy()):04d}_canonical.ply")
        mesh_deformed.export(f"test_mesh/{int(idx.cpu().numpy()):04d}_deformed.ply")
        self.model.deformer = SMPLDeformer(betas=np.load(self.betas_path), gender=self.gender)
        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                                min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"uv": inputs["uv"][:, indices],
                            "intrinsics": inputs['intrinsics'],
                            "frame_encoding_vector": inputs["frame_encoding_vector"],
                            "pose": inputs['pose'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:, 4:76],
                            "smpl_shape": inputs["smpl_params"][:, 76:],
                            "smpl_trans": inputs["smpl_params"][:, 1:4],
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

            body_model_params = self.body_model_params(inputs['idx'])

            batch_inputs.update({'smpl_pose': torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)})
            batch_inputs.update({'smpl_shape': body_model_params['betas']})
            batch_inputs.update({'smpl_trans': body_model_params['transl']})

            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                             "img_size": targets["img_size"]}

            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            results.append({"rgb_values":model_outputs["rgb_values"].detach().clone(), 
                            "fg_rgb_values":model_outputs["fg_rgb_values"].detach().clone(),
                            "normal_values": model_outputs["normal_values"].detach().clone(),
                            "acc_map": model_outputs["acc_map"].detach().clone(),
                            **batch_targets})         

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([result["fg_rgb_values"] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        pred_mask = torch.cat([result["acc_map"] for result in results], dim=0)
        pred_mask = pred_mask.reshape(*img_size, -1)

        if results[0]['rgb'] is not None:
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            rgb_gt = rgb_gt.reshape(*img_size, -1)
            rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        else:
            rgb = torch.cat([rgb_pred], dim=0).cpu().numpy()
        if 'normal' in results[0].keys():
            normal_gt = torch.cat([result["normal"] for result in results], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()
        
        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        cv2.imwrite(f"test_mask/{int(idx.cpu().numpy()):04d}.png", pred_mask.cpu().numpy() * 255)
        cv2.imwrite(f"test_rendering/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"test_fg_rendering/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])
        
        # Computing evaluation metrics ----------------------------------------------------------- 
        #rgb_pred_np = rgb_pred.permute(2, 0, 1).cpu().numpy()  # Convert from (H, W, C) to (C, H, W)
        #rgb_gt_np = rgb_gt.permute(2, 0, 1).cpu().numpy()
        print("rgb shapes:", rgb_pred.shape, rgb_gt.shape)
        rgb_pred_np_greysc = cv2.cvtColor(rgb_pred.cpu().numpy(), cv2.COLOR_BGR2GRAY)   
        rgb_gt_np_greysc = cv2.cvtColor(rgb_gt.cpu().numpy(), cv2.COLOR_BGR2GRAY)
        rgb_pred_np = rgb_pred.cpu().numpy() 
        rgb_gt_np = rgb_gt.cpu().numpy()
        print("images shapes:", rgb_pred_np.shape, rgb_gt_np.shape)
        #mask_multichannel = np.repeat(pred_mask.cpu().numpy(), 3, axis=0)
        print("mask shape:", pred_mask.shape)
        mask_rgb_pred_np_greysc = rgb_pred_np_greysc * pred_mask[:,:,0].cpu().numpy()
        mask_rgb_gt_np_greysc = rgb_gt_np_greysc * pred_mask[:,:,0].cpu().numpy()
        mask_rgb_pred_np = rgb_pred_np * pred_mask.cpu().numpy()
        mask_rgb_gt_np = rgb_gt_np * pred_mask.cpu().numpy()
        print("mask rgb shapes:", mask_rgb_pred_np.shape, mask_rgb_gt_np.shape)

        # Compute metrics
        validation_metrics = {
            f'ssim': compute_ssim(rgb_gt_np_greysc, rgb_pred_np_greysc),   # Normalized rgb values are passed
            f'psnr': compute_psnr(rgb_gt_np, rgb_pred_np),    # Normalized rgb values are passed
            f'mask_ssim': compute_ssim(mask_rgb_gt_np_greysc, mask_rgb_pred_np_greysc),   # Normalized rgb values are passed
            f'mask_psnr': compute_psnr(mask_rgb_gt_np, mask_rgb_pred_np)    # Normalized rgb values are passed
        }

        # Log the metrics values
        for k, v in validation_metrics.items():
            self.average_val_met[k] = self.average_val_met[k] + v
            print(f'{k}: {v}')

        return validation_metrics

    def test_epoch_end(self, outputs) -> None:
        for k,v in self.average_val_met.items():
            self.average_val_met[k] = self.average_val_met[k] / 42   # 42 is the number of frames. It needs to be changed if the test video is changed
            self.log(k, self.average_val_met[k], prog_bar=True, on_step=False)
        print(self.average_val_met)