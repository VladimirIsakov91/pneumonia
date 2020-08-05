from torch.utils.data import Dataset
import os
import numpy
from PIL import Image
from matplotlib import pyplot as plt


class ImageDataset(Dataset):

    def __init__(self, directory, labels):

        self.directory = directory
        self.labels = labels

    def __len__(self):
        return len(os.listdir(self.directory))

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.directory, item))
        image = numpy.array(image).astype(dtype=numpy.float32)
        return image, 0


if __name__ == '__main__':

    dataset = ImageDataset('/home/vladimir/MachineLearning/Datasets/chest_xray/train/NORMAL',  None)


