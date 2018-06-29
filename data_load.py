from torch.utils.data import Dataset
import os
import glob
from PIL import Image


class Cloth(Dataset):
    """
    Multi-label cloth dataset
    """

    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Directory with all sub-folders
        :param transform: Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = glob.glob(os.path.join(root_dir, '*/*'))
        self.color_classes = {'black': 0, 'blue': 1, 'red': 2}
        self.type_classes = {'dress': 0, 'jeans': 1, 'shirt': 2}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        path = self.image_names[item]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        targets = path.split('/')[-2].split('_')
        return img, self.color_classes[targets[0]], self.type_classes[targets[1]]


