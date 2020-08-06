from torch.utils.data import Dataset
import zarr


class ImageDataset(Dataset):

    def __init__(self, data):

        self.data = zarr.open(data, 'r')

    def __len__(self):
        return self.data['data'].shape[0]

    def __getitem__(self, item):

        return self.data['data'][item], self.data['labels'][item]


if __name__ == '__main__':

    pass



