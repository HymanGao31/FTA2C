import csv
import glob
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Traffic(Dataset):

    def __init__(self, root, resize, mode):
        super(Traffic, self).__init__()
        self.root = root
        self.resize = resize
        # 编码str的类型，使用dict映射。
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):  # 排序操作一下，避免每次加载尽量的顺序不一样。
            # 如果不是目录，过滤掉。
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # image, label

        self.images, self.labels = self.load_csv('traffic.csv')
        # 裁剪训练集和测试集 6:4
        if mode == 'train':  # 60%
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        elif mode == 'val':  # 20% = 60%->80%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def __len__(self):

        return len(self.images)  # 1167*0.6或者*0.2

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path => image data
            transforms.Resize((int(self.resize), int(self.resize))),
            transforms.RandomRotation(15),  # 随机旋转-15~15
            transforms.CenterCrop(self.resize),  # 中心裁剪成resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 这些数据是统计大量的imagnet来的。
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

    def load_csv(self, filename):
        # 如果不存在，才创建csv。
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'Traffic\\mewtwo\\00001.png，类别信息mewtwo使用此路径来判定。
                # 图片可能有多种格式，使用glob通配符
                images += glob.glob(os.path.join(self.root, name, '*.PNG'))
                images += glob.glob(os.path.join(self.root, name, '*.JPE'))
                images += glob.glob(os.path.join(self.root, name, '*.JPEG'))
            print(len(images), images)  # 1167, 'Traffic\\bulbasaur\\00000000.png'
            # 打乱图片
            random.shuffle(images)
            # 把路径和图片的关系，存储到csv，这样使得及时图片在任何位置都可以匹配到。
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'Traffic\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]  # bulbasaur
                    label = self.name2label[name]
                    writer.writerow([img, label])  # 'Traffic\\bulbasaur\\00000000.png', 0
                print('writen into csv file:', filename)

        # 存在直接加载csv
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row  # 'Traffic\\bulbasaur\\00000000.png', 0
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)  # 保证数据一致
        return images, labels
