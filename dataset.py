import os
import torch
from skimage import io, transform

import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np

class FaceLandmarksDataset(Dataset):
    """Face Landmarks Dataset
    

    :param Dataset: _description_
    :type Dataset: _type_
    """

    def __init__(self, csv_path, root_dir, transform=None):
        """Initializer of the Class

        :param csv_path: File Path where annotation and filename are mapped
        :type csv_path: str

        :param root_dir: Directory with All Images
        :type root_dir: str

        :param transform: Optional Transform to be Applied
        :type transform: _type_, optional
        """

        self.landmarks_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        """Retrieve 1 image for given index

        :param idx: _description_
        :type idx: _type_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])

        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # Torch Image: C x H x W

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
    

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        ## Original Height & Width
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            ## Say output size is 500 ##
            ## H: 434 | W: 290 ##
            ## New Size ##
            ## 748, 500 ##
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))

            ## Scaling the landmarks ##
            landmarks = landmarks * [new_w / w, new_h / h]

            return {'image': img, 'landmarks': landmarks}
    


    

transformed_dset = FaceLandmarksDataset(csv_path='data/faces/face_landmarks.csv',
                                        root_dir='data/faces/',
                                        transform=transforms.Compose([
                                            Rescale(256),
                                            RandomCrop(224),
                                            ToTensor()
                                        ]))

dataloader = DataLoader(transformed_dset, batch_size=4, shuffle=True, num_workers=)

if __name__ == "__main__":
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())
    

