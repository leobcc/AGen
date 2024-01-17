import os
import hydra
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class VideosDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = hydra.utils.to_absolute_path(data_dir)
        self.video_folders = [os.path.join(self.data_dir, folder) for folder in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, folder))]

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder_path = self.video_folders[idx]
        return video_folder_path

def create_video_dataloader(opt):
    # Create an instance of the dataset
    data_dir = opt.dataset_dir

    video_dataset = VideosDataset(data_dir)

    # Set up the data loader
    batch_size = opt.batch_size  
    shuffle = opt.shuffle   
    num_workers = opt.num_workers  

    data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader
