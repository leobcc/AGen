import torch.nn as nn
import torch
from torchvision import models
import torch.nn.init as init
import numpy as np
from .embedders import get_embedder
from lib.utils import utils

class ImplicitNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        dims = [opt.d_in] + list(
            opt.dims) + [opt.d_out + opt.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt

        # geometry encoding x sdf
        #self.geometry_encoding_net = geometry_encoding_net

        if opt.multires > 0:
            embed_fn, input_ch = get_embedder(opt.multires, input_dims=opt.d_in, mode=opt.embedder_mode)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.cond = opt.cond   
        if self.cond == 'smpl':
            self.cond_layer = [0]
            self.cond_dim = 69
        elif self.cond == 'frame':
            self.cond_layer = [0]
            self.cond_dim = opt.dim_frame_encoding
        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            
            if self.cond != 'none' and l in self.cond_layer:
                lin = nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)
            if opt.init == 'geometry':
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
            if opt.init == 'zero':
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    '''
    # geometry encoding x sdf
    def morphing_distances(self, input, cond, frame_encoding_vector, uv, morph=True, current_epoch=None):
        if input.ndim == 2: input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0: return input

        input = input.reshape(num_batch * num_point, num_dim)

        if input.shape[1] == 3 and morph != False:   # Foreground network
            deformed_input = self.geometry_encoding_net(input, frame_encoding_vector, uv)
            morphing_distances = deformed_input - input
        else:
            morphing_distances = None

        return morphing_distances
    '''

    def forward(self, input, cond, current_epoch=None):
        if input.ndim == 2: input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0: return input

        input = input.reshape(num_batch * num_point, num_dim)

        # geometry encoding x sdf
        #if input.shape[1] == 3 and morph != False:   # Foreground network
        #    input = self.geometry_encoding_net(input, frame_encoding_vector, uv)

        if self.cond != 'none':
            num_batch, num_cond = cond[self.cond].shape
            
            input_cond = cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)

            input_cond = input_cond.reshape(num_batch * num_point, num_cond)

            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        x = x.reshape(num_batch, num_point, -1)

        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.mode = opt.mode
        dims = [opt.d_in + opt.feature_vector_size] + list(
            opt.dims) + [opt.d_out]

        self.embedview_fn = None
        if opt.multires_view > 0:
            embedview_fn, input_ch = get_embedder(opt.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        if self.mode == 'nerf_frame_encoding':
            dims[0] += opt.dim_frame_encoding
        if self.mode == 'pose':
            self.dim_cond_embed = 8 
            self.cond_dim = 69 # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, points, normals, view_dirs, body_pose, feature_vectors, frame_latent_code=None):
        if self.embedview_fn is not None:
            if self.mode == 'nerf_frame_encoding':
                view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'nerf_frame_encoding':
            frame_latent_code = frame_latent_code.expand(view_dirs.shape[0], -1)
            rendering_input = torch.cat([view_dirs, frame_latent_code, feature_vectors], dim=-1)
        elif self.mode == 'pose':
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)
            rendering_input = torch.cat([points, normals, body_pose, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x

from .unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 32)
        self.down_convolution_2 = DownSample(32, 64)
        self.down_convolution_3 = DownSample(64, 128)
        #self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(128, 256)

        #self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(256, 128)
        self.up_convolution_3 = UpSample(128, 64)
        self.up_convolution_4 = UpSample(64, 32)
        
        self.out = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        #down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p3)

        #up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(b, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out

class GeometryEncodingNet(nn.Module):
    '''GeometryEncodingNet
    This network is used to encode the geometry of the scene in a latent space. 
    It morphs the canonical space into one of which the geometry is encoded in the latent space.
    It takes as input the coordinates of the points and the encoding vector of the frame.
    It process the encoding vector of the frame and concatenate it with the points' coordinates, 
    to then process them together and output the coordinates of the points in the latent space.
    '''
    def __init__(self, n_x_c=3, n_encoding_vector=512, in_channels=3, out_channels=16, hidden_size=256, output_size=3):
        super(GeometryEncodingNet, self).__init__()

        self.morphing_factor = 0.1   # 0.1

        # Unet to process the frame
        self.unet = UNet(in_channels, out_channels)

        # Define fully connected layers to process the points with the hidden encoding vector
        self.fc1 = nn.Linear(n_x_c + out_channels, n_x_c + out_channels)
        self.fc2 = nn.Linear(n_x_c + out_channels, n_x_c + out_channels)
        self.fc3 = nn.Linear(n_x_c + out_channels, n_x_c + out_channels)
        self.fc4 = nn.Linear(n_x_c + out_channels, output_size)

    def forward(self, x_c, frame_encoding_vector, uv):
        # Encoder
        frame_encoding_vector = self.unet(frame_encoding_vector)

        # Concatenate the frame encoding vector with the points coordinates
        # TODO: concat with encoding of respective rays
        encodings = utils.get_uv_rgb_values(frame_encoding_vector, uv)
        # Repeat the encoding vector along a new dimension to match the number of points
        # The repeated dimension should correspond to the number of points per ray
        repeated_encoding = encodings.unsqueeze(0).repeat(x_c.shape[0] // encodings.shape[1], 1, 1).permute(2, 0, 1)
        # Concatenate the repeated encoding vector to x_c
        x = torch.cat((x_c, repeated_encoding.reshape(-1, repeated_encoding.shape[2])), dim=-1)

        # Injects the frame encoding vector in the points' coordinates
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        output = x_c + (x - x_c)*self.morphing_factor

        return output

class UNet_V4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 32)
        self.down_convolution_2 = DownSample(32, 64)
        self.down_convolution_3 = DownSample(64, 128)
        #self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(128, 256)

        #self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(256, 128)
        self.up_convolution_3 = UpSample(128, 64)
        self.up_convolution_4 = UpSample(64, 32)
        
        self.out = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        #down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p3)

        #up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(b, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out

class GeometryEncodingNet_V4(nn.Module):
    '''GeometryEncodingNet
    This network is used to encode the geometry of the scene in a latent space. 
    It morphs the canonical space into one of which the geometry is encoded in the latent space.
    It takes as input the coordinates of the points and the encoding vector of the frame.
    It process the encoding vector of the frame and concatenate it with the points' coordinates, 
    to then process them together and output the coordinates of the points in the latent space.
    '''
    def __init__(self, n_x_c=3, n_encoding_vector=512, in_channels=3, out_channels=16, hidden_size=256, output_size=3):
        super(GeometryEncodingNet, self).__init__()
        self.morphing_factor = 0.1

        # Unet to process the frame
        self.unet = UNet(in_channels, out_channels)

        # Define fully connected layers to process the points with the hidden encoding vector
        self.fc1 = nn.Linear(n_x_c + out_channels, n_x_c + out_channels)
        self.fc2 = nn.Linear(n_x_c + out_channels, n_x_c + out_channels)
        self.fc3 = nn.Linear(n_x_c + out_channels, n_x_c + out_channels)
        self.fc4 = nn.Linear(n_x_c + out_channels, output_size)

    def forward(self, x_c, frame_encoding_vector, uv):
        # Encoder
        frame_encoding_vector = self.unet(frame_encoding_vector)

        # Concatenate the frame encoding vector with the points coordinates
        # TODO: concat with encoding of respective rays
        encodings = utils.get_uv_rgb_values(frame_encoding_vector, uv)
        # Repeat the encoding vector along a new dimension to match the number of points
        # The repeated dimension should correspond to the number of points per ray
        repeated_encoding = encodings.unsqueeze(0).repeat(x_c.shape[0] // encodings.shape[1], 1, 1).permute(2, 0, 1)
        # Concatenate the repeated encoding vector to x_c
        x = torch.cat((x_c, repeated_encoding.reshape(-1, repeated_encoding.shape[2])), dim=-1)

        # Injects the frame encoding vector in the points' coordinates
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        output = x_c + (x - x_c)*self.morphing_factor

        return output

class GeometryEncodingNet_V3(nn.Module):
    '''GeometryEncodingNet
    This network is used to encode the geometry of the scene in a latent space. 
    It morphs the canonical space into one of which the geometry is encoded in the latent space.
    It takes as input the coordinates of the points and the encoding vector of the frame.
    It process the encoding vector of the frame and concatenate it with the points' coordinates, 
    to then process them together and output the coordinates of the points in the latent space.
    '''
    def __init__(self, n_x_c=3, n_encoding_vector=512, hidden_size=256, hidden_encoding_size=256, output_size=3):
        super(GeometryEncodingNet, self).__init__()
        self.morphing_factor = 0.5

        # Define fully connected layers to process the encoding vector of the frame
        self.fc1_e = nn.Linear(n_encoding_vector, hidden_encoding_size)
        #self.fc2_e = nn.Linear(hidden_size, hidden_encoding_size)
        #self.fc3_e = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        #self.fc4_e = nn.Linear(int(hidden_size/4), hidden_encoding_size)

        # Define fully connected layers to process the points with the hidden encoding vector
        self.fc1 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc2 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc3 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc4 = nn.Linear(hidden_encoding_size+n_x_c, output_size)

        # Initialize weights close to the identity function
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a small amount of noise around zero
                init.normal_(m.weight, mean=1, std=0.01)
                init.constant_(m.bias, 0)

    def forward(self, x_c, frame_encoding_vector):
        # Process the encoding vector of the frame
        frame_encoding_vector = self.fc1_e(frame_encoding_vector)
        #frame_encoding_vector = self.fc2_e(frame_encoding_vector)
        #frame_encoding_vector = torch.relu(self.fc3_e(frame_encoding_vector))
        #frame_encoding_vector = self.fc4_e(frame_encoding_vector)

        # Concatenate the frame encoding vector with the points coordinates
        x = torch.cat((x_c, frame_encoding_vector.expand(x_c.shape[0], -1)), dim=1)

        # Injects the frame encoding vector in the points' coordinates
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        output = x_c + (x - x_c)*self.morphing_factor

        return output

class GeometryEncodingNet_V2(nn.Module):
    '''GeometryEncodingNet
    This network is used to encode the geometry of the scene in a latent space. 
    It morphs the canonical space into one of which the geometry is encoded in the latent space.
    It takes as input the coordinates of the points and the encoding vector of the frame.
    It process the encoding vector of the frame and concatenate it with the points' coordinates, 
    to then process them together and output the coordinates of the points in the latent space.
    '''
    def __init__(self, n_x_c=3, n_encoding_vector=512, hidden_size=256, hidden_encoding_size=256, output_size=3):
        super(GeometryEncodingNet, self).__init__()
        self.morphing_factor = 0.1

        # Define fully connected layers to process the encoding vector of the frame
        self.fc1_e = nn.Linear(n_encoding_vector, hidden_size)
        self.fc2_e = nn.Linear(hidden_size, hidden_encoding_size)
        #self.fc3_e = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        #self.fc4_e = nn.Linear(int(hidden_size/4), hidden_encoding_size)

        # Define fully connected layers to process the points with the hidden encoding vector
        self.fc1 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc2 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc3 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc4 = nn.Linear(hidden_encoding_size+n_x_c, output_size)

        # Initialize weights close to the identity function
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a small amount of noise around zero
                init.normal_(m.weight, mean=1, std=0.01)
                init.constant_(m.bias, 0)

    def forward(self, x_c, frame_encoding_vector):
        # Process the encoding vector of the frame
        frame_encoding_vector = torch.relu(self.fc1_e(frame_encoding_vector))
        frame_encoding_vector = self.fc2_e(frame_encoding_vector)
        #frame_encoding_vector = torch.relu(self.fc3_e(frame_encoding_vector))
        #frame_encoding_vector = self.fc4_e(frame_encoding_vector)

        # Concatenate the frame encoding vector with the points coordinates
        x = torch.cat((x_c, frame_encoding_vector.expand(x_c.shape[0], -1)), dim=1)

        # Injects the frame encoding vector in the points' coordinates
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        output = x_c + (x - x_c)*self.morphing_factor

        return output

class GeometryEncodingNet_V1(nn.Module):
    '''GeometryEncodingNet
    This network is used to encode the geometry of the scene in a latent space. 
    It morphs the canonical space into one of which the geometry is encoded in the latent space.
    It takes as input the coordinates of the points and the encoding vector of the frame.
    It process the encoding vector of the frame and concatenate it with the points' coordinates, 
    to then process them together and output the coordinates of the points in the latent space.
    '''
    def __init__(self, n_x_c=3, n_encoding_vector=512, hidden_size=256, hidden_encoding_size=256, output_size=3):
        super(GeometryEncodingNet, self).__init__()
        self.morphing_factor = 0.01

        # Define fully connected layers to process the encoding vector of the frame
        self.fc1_e = nn.Linear(n_encoding_vector, hidden_size)
        self.fc2_e = nn.Linear(hidden_size, hidden_encoding_size)
        #self.fc3_e = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        #self.fc4_e = nn.Linear(int(hidden_size/4), hidden_encoding_size)

        # Define fully connected layers to process the points with the hidden encoding vector
        self.fc1 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc2 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc3 = nn.Linear(hidden_encoding_size+n_x_c, hidden_encoding_size+n_x_c)
        self.fc4 = nn.Linear(hidden_encoding_size+n_x_c, output_size)

        # Initialize weights close to the identity function
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a small amount of noise around zero
                init.normal_(m.weight, mean=1, std=0.01)
                init.constant_(m.bias, 0)

    def forward(self, x_c, frame_encoding_vector):
        # Process the encoding vector of the frame
        frame_encoding_vector = torch.relu(self.fc1_e(frame_encoding_vector))
        frame_encoding_vector = self.fc2_e(frame_encoding_vector)
        #frame_encoding_vector = torch.relu(self.fc3_e(frame_encoding_vector))
        #frame_encoding_vector = self.fc4_e(frame_encoding_vector)

        # Concatenate the frame encoding vector with the points coordinates
        x = torch.cat((x_c, frame_encoding_vector.expand(x_c.shape[0], -1)), dim=1)

        # Injects the frame encoding vector in the points' coordinates
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        output = x_c + (x - x_c)*self.morphing_factor

        return output
