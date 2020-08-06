from torch.utils.data import Dataset
import os
import numpy
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, directory, labels):

        self.directory = directory
        self.labels = labels
        self.files = os.listdir(self.directory)

        self._prepare_data()

    def _prepare_data(self):

        images = (Image.open(os.path.join(self.directory, file)).resize((64, 64)) for file in self.files)
        self.data = [numpy.expand_dims(numpy.array(image).astype(dtype=numpy.float32), axis=0) for image in images]
        self.data = numpy.stack(self.data, axis=0)

    def __len__(self):
        return len(os.listdir(self.directory))

    def __getitem__(self, item):

        return self.data[item], 0


if __name__ == '__main__':

    pass



