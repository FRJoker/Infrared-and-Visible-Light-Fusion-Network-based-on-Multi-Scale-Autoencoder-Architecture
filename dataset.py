import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy


class MyDataLoader(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_list = []

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # 根据索引获取单个样本
        sample = self.image_list[index]
        return sample

    def load_images_from_folder(self):

        for filename in os.listdir(self.folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.folder_path, filename)
                image = self.load_image(image_path)
                self.image_list.append(image)
        return self.image_list

    def load_image(self, image_path):
        image = Image.open(image_path).convert('L')
        image = image.resize((256, 256), resample=1)
        image = numpy.array(image)
        image = numpy.reshape(image, [1, image.shape[0], image.shape[1]])
        # image_tensor = transforms.ToTensor()(image)
        image_tensor = torch.from_numpy(image).float()
        return image_tensor


