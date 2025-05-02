from typing import Any
from torch.utils.data import Dataset
import glob
from PIL import Image

class TrainTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None) -> None:
        super().__init__()
        self.filenames = glob.glob(root + "/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: Any) -> Any:
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.id_dict[img_path.split('/')[-3]]
        if self.transform:
            image = self.transform(image)
        return image, label


class ValTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(root + "/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(root + '/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label

