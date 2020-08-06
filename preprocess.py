import zarr
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import os
from PIL import Image
import numpy


class Preprocessor:

    def __init__(self):
        pass

    @dask.delayed
    def read_image(self, file, resize):

        image = Image.open(file).convert('L')

        if resize:
            image = image.resize(resize)

        image = numpy.array(image)

        return image

    def _preprocess(self, collection, chunks=64, size=None):

        h, w = size
        images = [self.read_image(file, (h, w)) for file in collection]
        images = [da.from_delayed(image, shape=(h, w), dtype=numpy.uint8) for image in images]
        images = da.stack(images, axis=0)
        images = images.rechunk(chunks=(chunks, h, w))

        return images

    def _to_zarr(self, data, labels, location):

        data = da.to_zarr(data, location, component='data', compute=False)
        labels = da.to_zarr(labels, location, component='labels', compute=False)

        return data, labels

    def preprocess(self, collection, labels, location, chunks=64, size=None):

        data = self._preprocess(collection, chunks, size)
        labels = da.from_array(labels, chunks=(chunks, ))
        data, labels = self._to_zarr(data, labels, location)
        dask.compute([data, labels])


def get_data(index):
    files = os.listdir(index)
    out = [os.path.join(index, file) for file in files]
    return out


if __name__ == '__main__':

    cluster = LocalCluster(n_workers=10)
    client = Client(cluster)

    p = Preprocessor()

    normal = get_data('/home/vladimir/MachineLearning/Datasets/chest_xray/test/NORMAL')
    pneumonia = get_data('/home/vladimir/MachineLearning/Datasets/chest_xray/test/PNEUMONIA')
    n_labels = [0 for _ in range(len(normal))]
    p_labels = [1 for _ in range(len(pneumonia))]

    data = normal + pneumonia
    labels = n_labels + p_labels

    labels = numpy.array(labels).astype(dtype=numpy.int64)

    p.preprocess(collection=data,
                 location='./test.zarr',
                 chunks=256,
                 size=(64, 64),
                 labels=labels)


