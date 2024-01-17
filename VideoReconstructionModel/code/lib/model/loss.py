import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = opt.eikonal_weight
        self.bce_weight = opt.bce_weight
        self.opacity_sparse_weight = opt.opacity_sparse_weight
        self.in_shape_weight = opt.in_shape_weight
        #self.norm_reg_weight = opt.norm_reg_weight 
        #self.time_cons_weight = opt.time_cons_weight
        #self.sdf_soft_time_cons_weight = opt.sdf_soft_time_cons_weight
        self.geometry_morphing_weight = opt.geometry_morphing_weight
        self.eps = 1e-6
        self.milestone = 200
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
    
    # L1 reconstruction loss for RGB values
    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    # Eikonal loss introduced in IGR
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss

    # BCE loss for clear boundary
    def get_bce_loss(self, acc_map):
        binary_loss = -1 * (acc_map * (acc_map + self.eps).log() + (1-acc_map) * (1 - acc_map + self.eps).log()).mean() * 2
        return binary_loss

    # Global opacity sparseness regularization 
    def get_opacity_sparse_loss(self, acc_map, index_off_surface):
        opacity_sparse_loss = self.l1_loss(acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface]))
        return opacity_sparse_loss

    # Optional: This loss helps to stablize the training in the very beginning
    def get_in_shape_loss(self, acc_map, index_in_surface):
        in_shape_loss = self.l1_loss(acc_map[index_in_surface], torch.ones_like(acc_map[index_in_surface]))
        return in_shape_loss
    
    # Optional: Normal regularization loss (regularization of the sdf normals on the surface)
    def get_norm_reg_loss(self, normal_values):
        squared_norm = (torch.sum(normal_values * normal_values.roll(1, 0), dim=-1).unsqueeze(-1))** 2
        norm_reg_loss = torch.mean(1 - squared_norm)
        return norm_reg_loss

    # Optional: Time consistency loss (imposes time consistency on the canonical representation)
    def get_time_cons_loss(self, canonical_surface_points, previous_canonical_surface_points):
        #time_cons_loss = torch.mean(torch.sqrt(torch.sum((canonical_surface_points*previous_canonical_surface_points)**2, dim=-1).unsqueeze(-1)))
        distances = torch.cdist(canonical_surface_points, previous_canonical_surface_points, p=2)
        min_distances, _ = torch.min(distances, dim=1)
        time_cons_loss = torch.sum(min_distances)
        return time_cons_loss

    # Optional: sdf soft time consistency loss (imposes time consistency on the canonical representation by softly regularizing the signed distance function)
    def get_sdf_soft_time_cons_loss(self, previous_sdf_values, sdf_values_on_previous_points):
        sdf_soft_time_cons_loss = F.mse_loss(sdf_values_on_previous_points, previous_sdf_values)
        return sdf_soft_time_cons_loss
    
    def get_sdf_soft_increment_epoch(self, epoch):
        if epoch<1000:
            return 1
        elif epoch<1100:
            return (1100-epoch)*10
        else:
            return 10e3 

    def get_geometry_morphing_loss(self, morphing_distances):
        #geometry_morphing_loss = torch.mean(morphing_distances**2)
        return morphing_distances

    def forward(self, model_outputs, ground_truth):
        nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        rgb_gt = ground_truth['rgb'][0].cuda()
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'][nan_filter], rgb_gt[nan_filter])
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        bce_loss = self.get_bce_loss(model_outputs['acc_map'])
        opacity_sparse_loss = self.get_opacity_sparse_loss(model_outputs['acc_map'], model_outputs['index_off_surface'])
        in_shape_loss = self.get_in_shape_loss(model_outputs['acc_map'], model_outputs['index_in_surface'])
        #norm_reg_loss = self.get_norm_reg_loss(model_outputs['normal_values'])
        #time_cons_loss = self.get_time_cons_loss(model_outputs['canonical_surface_points'], model_outputs['previous_canonical_surface_points'])
        #sdf_soft_time_cons_loss = self.get_sdf_soft_time_cons_loss(model_outputs['previous_sdf_values'], model_outputs['sdf_values_on_previous_points'])
        geometry_morphing_loss = self.get_geometry_morphing_loss(model_outputs['morphing_distances'])
        curr_epoch_for_loss = min(self.milestone, model_outputs['epoch']) # will not increase after the milestone

        loss = rgb_loss + \
                self.eikonal_weight * eikonal_loss + \
                self.bce_weight * bce_loss + \
                self.opacity_sparse_weight * (1 + curr_epoch_for_loss ** 2 / 40) * opacity_sparse_loss + \
                self.in_shape_weight * (1 - curr_epoch_for_loss / self.milestone) * in_shape_loss + \
                self.geometry_morphing_weight * geometry_morphing_loss
                # self.sdf_soft_time_cons_weight *  (1+curr_epoch_for_loss) * sdf_soft_time_cons_loss
                # self.time_cons_weight * (1 + curr_epoch_for_loss) * time_cons_loss
                # self.norm_reg_weight * (norm_reg_loss)

        
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'bce_loss': bce_loss,
            'opacity_sparse_loss': opacity_sparse_loss,
            'in_shape_loss': in_shape_loss,
            'geometry_morphing_loss': geometry_morphing_loss,
        }