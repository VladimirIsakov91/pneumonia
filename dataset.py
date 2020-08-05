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

        image = Image.open(os.path.join(self.directory, item)).resize((64, 64))
        image = numpy.array(image).astype(dtype=numpy.float32)
        image = image[numpy.newaxis, :, :]

        return image, 0


if __name__ == '__main__':

    dataset = ImageDataset('/home/vladimir/MachineLearning/Datasets/chest_xray/train/NORMAL',  None)
    image = dataset['IM-0115-0001.jpeg'][0]
    print(image.shape)


