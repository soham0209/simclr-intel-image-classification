
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
import torch
from PIL import Image 
import os

class IntelDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, split, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            split (string): train/val/test
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split = split
        self.split_lists = ['seg_train', 'seg_test', 'seg_pred']
        self.root_dir = root
        self.transform = transform
        self.file_names_frame = pd.read_csv(os.path.join(root, split + '.csv'))
        

    def __len__(self):
        return len(self.file_names_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.split, 
                                self.file_names_frame.loc[idx, 'image'])
        image = Image.open(img_name)
        label = self.file_names_frame.loc[idx, 'label']
        # label = np.array([label], dtype=float).reshape(-1, 1)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def preprocess_dataset(root_dir, split):
    cat_map = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
    df = pd.DataFrame(columns=['image', 'label'])
    data = {'image': [], 'label': []}
    if split in ['seg_train', 'seg_test']:    
        for cat in cat_map.keys():
            cat_dir = os.path.join(root_dir, split, cat)
            for img in os.listdir(cat_dir):
                data['image'].append(os.path.join(cat, img))
                data['label'].append(cat_map[cat])
    else:
        for img in os.listdir(os.path.join(root_dir, split)):
            data['image'].append(img)
            data['label'].append(-1)
    df = pd.DataFrame(data, columns=['image', 'label'])
    df.to_csv(os.path.join(root_dir, split + '.csv'), index=True)

if __name__ == '__main__':
    dataset = IntelDataset(root='intel-classification-dataset', split='seg_train')
    print(dataset[0])