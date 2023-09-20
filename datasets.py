import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        if not root.endswith("tiny-imagenet-200"):
            root = os.path.join(root, "tiny-imagenet-200")
        self.train_dir = os.path.join(root, "train")
        self.val_dir = os.path.join(root, "val")
        self.transform = transform
        if train:
            self._scan_train()
        else:
            self._scan_val()

    def _scan_train(self):
        classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        classes = sorted(classes)
        assert len(classes) == 200

        self.data = []
        for idx, name in enumerate(classes):
            this_dir = os.path.join(self.train_dir, name)
            for root, _, files in sorted(os.walk(this_dir)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        item = (path, idx)
                        self.data.append(item)
        self.labels_dict = {i: classes[i] for i in range(len(classes))}

    def _scan_val(self):
        self.file_to_class = {}
        classes = set()
        with open(os.path.join(self.val_dir, "val_annotations.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            words = line.split("\t")
            self.file_to_class[words[0]] = words[1]
            classes.add(words[1])
        classes = sorted(list(classes))
        assert len(classes) == 200

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.data = []
        this_dir = os.path.join(self.val_dir, "images")
        for root, _, files in sorted(os.walk(this_dir)):
            for fname in sorted(files):
                if fname.endswith(".JPEG"):
                    path = os.path.join(root, fname)
                    idx = class_to_idx[self.file_to_class[fname]]
                    item = (path, idx)
                    self.data.append(item)
        self.labels_dict = {i: classes[i] for i in range(len(classes))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataset(name, root="./data", train=True, flip=False, crop=False):
    if name == 'cifar':
        DATASET = CIFAR10
        RES = 32
    elif name == 'tiny':
        DATASET = TinyImageNet
        RES = 64
    else:
        raise NotImplementedError

    tf = [transforms.ToTensor()]
    if train:
        if crop:
            tf = [transforms.RandomCrop(RES, 4)] + tf
        if flip:
            tf = [transforms.RandomHorizontalFlip()] + tf

    return DATASET(root=root, train=train, transform=transforms.Compose(tf))
