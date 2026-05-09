import os
import torch
from shutil import move, rmtree
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from shutil import copy2  # 导入复制函数
import numpy as np
from torchvision import transforms


def split_images_labels(imgs):
        images = []
        labels = []
        for item in imgs:
            images.append(item[0])
            labels.append(item[1])
        return np.array(images), np.array(labels)

class dataset(torch.utils.data.Dataset):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, root, split_ratio=0.8, dataset_name='OfficeHome'):
        self.root = os.path.expanduser(root)
        self.dataset_name = dataset_name

        self.fpath = os.path.join(self.root, dataset_name)
        if not os.path.exists(self.fpath):
            raise RuntimeError(f"Dataset path {self.fpath} does not exist.")

        self.train_folder = os.path.join(self.fpath, 'train')
        self.test_folder = os.path.join(self.fpath, 'test')

        if not os.path.exists(self.train_folder) or not os.path.exists(self.test_folder):
            self._split_domains(split_ratio)
        
        self.train_data = self._load_data_per_domain(self.train_folder)
        self.test_data = self._load_data_per_domain(self.test_folder)

        self._train_data, self._train_targets = self._split_data_targets(self.train_data)
        self._test_data, self._test_targets = self._split_data_targets(self.test_data)

        self.classes = self.test_data[next(iter(self.train_data))].classes
        self.domains = self._train_data.keys()

    def _split_data_targets(self, data_dict):
        images_dict = {}
        labels_dict = {}
        for domain, image_folder in data_dict.items():
            images, labels = split_images_labels(image_folder.imgs)
            images_dict[domain] = images
            labels_dict[domain] = labels
        return images_dict, labels_dict

    def _split_domains(self, split_ratio):
        if os.path.exists(self.train_folder):
            rmtree(self.train_folder)
        if os.path.exists(self.test_folder):
            rmtree(self.test_folder)
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)

        domains = [d for d in os.listdir(self.fpath) if d not in ["train", "test"]]
        for domain in domains:
            print(f"Processing domain: {domain}")
            domain_path = os.path.join(self.fpath, domain)
            if not os.path.isdir(domain_path):
                continue

            class_folders = [c for c in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, c))]
            for class_name in tqdm(class_folders, desc=f"  {domain}", unit="class", leave=False):
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                images = [img for img in os.listdir(class_path) 
                          if img.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if not images:
                    print(f"Warning: No valid images in {class_path}, the class is{class_name}.")
                    continue

                train_size = int(split_ratio * len(images))
                train_files = images[:train_size]
                test_files = images[train_size:]

                train_class_folder = os.path.join(self.train_folder, domain, class_name)
                test_class_folder = os.path.join(self.test_folder, domain, class_name)
                os.makedirs(train_class_folder, exist_ok=True)
                os.makedirs(test_class_folder, exist_ok=True)

                for img in train_files:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(train_class_folder, img)
                    copy2(src, dst)
                for img in test_files:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(test_class_folder, img)
                    copy2(src, dst)

            print(f"Processed domain: {domain}")

    def _load_data_per_domain(self, data_folder):
        domain_data = {}
        for domain in os.listdir(data_folder):
            domain_path = os.path.join(data_folder, domain)
            if not os.path.isdir(domain_path):
                continue
            print(f"Loading domain: {domain}")
            domain_data[domain] = ImageFolder(domain_path)
        return domain_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]