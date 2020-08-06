from torch.utils.data import Dataset
import zarr
import numpy


class ImageDataset(Dataset):

    def __init__(self, data, transform=None):

        self.data = zarr.open(data, 'r')
        self.transform = transform

    def __len__(self):
        return self.data['data'].shape[0]

    def __getitem__(self, item):

        image = self.data['data'][item]

        if self.transform:
            image = self.transform(image=image)['image']

        return image[numpy.newaxis, ...].astype(numpy.float32), self.data['labels'][item]


if __name__ == '__main__':

    pass



