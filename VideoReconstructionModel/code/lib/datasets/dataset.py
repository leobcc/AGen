import os
import glob
import hydra
import cv2
import numpy as np
import torch
from lib.utils import utils

from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split, pretrained_uv=None):
        root = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), 'data', metainfo.data_dir))

        self.start_frame = metainfo.start_frame
        self.end_frame = metainfo.end_frame 
        self.skip_step = 1
        self.images, self.img_sizes = [], []
        self.training_indices = list(range(metainfo.start_frame, metainfo.end_frame, self.skip_step))
        
        # images
        img_dir = os.path.join(root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))

        # only store the image paths to avoid OOM
        self.img_paths = [self.img_paths[i] for i in self.training_indices]
        for img_path in self.img_paths:
            if not os.path.exists(img_path):
                print(f'!!! The path {img_path} does not exist. !!!')            
        self.img_size = cv2.imread(self.img_paths[0]).shape[:2]
        self.n_images = len(self.img_paths)

        # coarse projected SMPL masks, only for sampling
        mask_dir = os.path.join(root, "mask")
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.mask_paths = [self.mask_paths[i] for i in self.training_indices]

        self.shape = np.load(os.path.join(root, "mean_shape.npy"))
        self.poses = np.load(os.path.join(root, 'poses.npy'))[self.training_indices]
        self.trans = np.load(os.path.join(root, 'normalize_trans.npy'))[self.training_indices]

        # Frame encoding vectors from pretrained encoder
        #self.frame_encoding_vector = np.load(os.path.join(root, 'feature_encoding_vectors.npy'))

        # cameras
        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.training_indices]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.training_indices]

        self.scale = 1 / scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = utils.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        assert len(self.intrinsics_all) == len(self.pose_all)

        # other properties
        self.num_sample = split.num_sample
        self.sampling_strategy = "weighted"

        self.sampling = split.sampling
        self.pretrained_uv = pretrained_uv

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # normalize RGB
        img = cv2.imread(self.img_paths[idx])

        # preprocess: BGR -> RGB -> Normalize
        img = img[:, :, ::-1] / 255

        mask = cv2.imread(self.mask_paths[idx])
        # preprocess: BGR -> Gray -> Mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0

        img_size = self.img_size

        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float() 

        smpl_params[1:4] = torch.from_numpy(self.trans[idx]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx]).float()
        smpl_params[76:] = torch.from_numpy(self.shape).float()

        # Pretrained encoding vector
        #self.frame_encoding_backbone.eval()
        #frame = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        #normalized_img = transforms.functional.normalize(frame, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #print(normalized_img.device)
        #frame_encoding_vector = self.frame_encoding_backbone(normalized_img).cpu()
        #print(frame_encoding_vector.shape)
        
        #frame_encoding_vector = self.frame_encoding_vector[idx]

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "object_mask": mask,
            }
            
            if self.pretrained_uv is not None:
                samples = {"uv": self.pretrained_uv[idx], 
                            "rgb": utils.rgb_from_pretrained_uv(self.pretrained_uv[idx], img)}
                index_outside = np.zeros(self.pretrained_uv[idx].shape[0])
                #print("samples[uv] type:", type(samples["uv"]))
                #print("samples[rgb] type:", type(["rgb"]))
                #print("index_outside type:", type(index_outside))
            else:
                samples, index_outside = utils.weighted_sampling(data, img_size, self.num_sample)
            inputs = {
                "uv": samples["uv"].astype(np.float32),
                "uv_0": np.copy(samples["uv"].astype(np.float32)),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                'index_outside': index_outside,
                "frame_encoding_vector": np.transpose(img, (2, 0, 1)).astype(np.float32),
                "idx": idx
            }
            images = {"rgb": samples["rgb"].astype(np.float32)}
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                "frame_encoding_vector": np.transpose(img, (2, 0, 1)).astype(np.float32),
                "idx": idx
            }
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32),
                "img_size": self.img_size
            }
            return inputs, images

# TODO: At the moment the validset creates an instance of the validation set but using the training set as basis. We need to change to use the videos in the validset
class ValDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split, pretrained_uv=None):
        self.dataset = Dataset(metainfo, split, pretrained_uv)
        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))  
        self.data = self.dataset[image_id]
        inputs, images = self.data

        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            "smpl_params": inputs["smpl_params"],
            'image_id': image_id,
            "frame_encoding_vector": inputs["frame_encoding_vector"],
            "idx": inputs['idx']
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"],
            'pixel_per_batch': self.pixel_per_batch,
            'total_pixels': self.total_pixels
        }
        return inputs, images

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split, pretrained_uv=None):
        self.dataset = Dataset(metainfo, split, pretrained_uv)

        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        inputs, images = data
        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            "smpl_params": inputs["smpl_params"],
            "frame_encoding_vector": inputs["frame_encoding_vector"],
            "idx": inputs['idx']
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"]
        }
        return inputs, images, self.pixel_per_batch, self.total_pixels, idx
