import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import os
import random
import numpy as np

class SiameseTripletNetwork(nn.Module):
    def __init__(self):
        super(SiameseTripletNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            #nn.Flatten()
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 128)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9216, 512),
            #nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output


    def forward(self, x):
        output = self.forward_once(x)

        return output


class SiameseTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SiameseTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class OneShotSiameseDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, num_samples_per_class=2):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.num_samples_per_class = num_samples_per_class
        self.image_paths = []
        self.labels = []

        for i, class_dir in enumerate(classes):
            class_image_paths = os.listdir(os.path.join(root_dir, class_dir))
            class_image_paths = [os.path.join(class_dir, p) for p in class_image_paths]
            self.image_paths.extend(class_image_paths[:num_samples_per_class])
            self.labels.extend([i] * min(num_samples_per_class, len(class_image_paths)))

    def __getitem__(self, index):
        anchor_path = self.image_paths[index]
        anchor_label = self.labels[index]

        # Select a random positive sample from the same class
        positive_indices = np.where(np.array(self.labels) == anchor_label)[0]
        positive_index = random.choice(positive_indices)
        positive_path = self.image_paths[positive_index]

        # Select a random negative sample from a different class
        negative_indices = np.where(np.array(self.labels) != anchor_label)[0]
        negative_index = random.choice(negative_indices)
        negative_path = self.image_paths[negative_index]

        anchor_image = self.load_image(anchor_path)
        positive_image = self.load_image(positive_path)
        negative_image = self.load_image(negative_path)

        return anchor_image, positive_image, negative_image, anchor_label

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, path):
        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image

