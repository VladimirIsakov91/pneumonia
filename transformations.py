import albumentations as A

train_tr = A.Compose([
    A.CLAHE(p=0.2),
    A.Flip(p=0.3),
    A.OneOf([A.GaussianBlur(),
             A.MedianBlur(),
             A.GaussNoise()], p=0.2),
    A.OpticalDistortion(p=0.2),
    A.OneOf([
            A.IAASharpen(),
            A.RandomBrightnessContrast()], p=0.2)
])

val_tr = A.Compose([
])
