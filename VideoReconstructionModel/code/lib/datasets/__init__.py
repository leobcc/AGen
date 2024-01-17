from .dataset import Dataset, ValDataset, TestDataset
from torch.utils.data import DataLoader, Subset

def find_dataset_using_name(name):
    mapping = {
        "VideoTrain": Dataset,
        "VideoVal": ValDataset,
        "VideoTest": TestDataset,
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}") 
    return cls


def create_dataset(metainfo, split, pretrained_uv=None):
    dataset_cls = find_dataset_using_name(split.type)
    dataset = dataset_cls(metainfo, split, pretrained_uv)
    if split.type == "VideoTest":
        if split.size == "reduced":
            dataset = Subset(dataset, split.frame_idxs)
    return DataLoader(
        dataset,
        batch_size=split.batch_size,
        drop_last=split.drop_last,
        shuffle=split.shuffle,
        num_workers=split.worker,
        pin_memory=True
    )