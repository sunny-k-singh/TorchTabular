import torch
import torchvision.transforms as tf
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, data, target, data_transform=None, target_transform=None):
        """
        This will take data labels and target labels to prepare it for 
        using inisde DataLoaders from Pytorch. This will also implement transform
        methods seperately for data and targets
        """

        if len(data) != len(target):
            raise ValueError("Data and Targets not of the same dimensions")

        super().__init__()
        self.data = data
        self.target = target
        self.data_transform = data_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        This will return the data at a given index or row, after transformations
        has been imlemented.
        """

        index_data = self.data[index]
        index_target = self.target[index]

        if self.data_transform is not None:

            try:
                index_data = self.data_transform(index_data)
            except:
                raise TypeError("This transform is not valid for your data")

        if self.target_transform is not None:

            try:
                index_target = self.data_transform(index_target)
            except:
                raise TypeError("This transform is not valid for your target")

        return index_data, index_target

    def __len__(self):
        """
        This simply gives the length of the data
        """

        return len(self.data)


if __name__ == "__main__":
    """
    This is for testing if it works in all scenarios imaginable
    """

    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 1)

    randomDataset = CustomDataset(X, Y)

    print("### 3rd index ###")
    print("original dataset: ", X[3], Y[3])
    print("transformed dataset: ", randomDataset[3])

    def toTensor(data):
        return torch.from_numpy(data)

    transformedDataset = CustomDataset(
        X, Y, data_transform=toTensor, target_transform=toTensor)

    print("### Transformed into Pytorch Tensors ###")
    print("original dataset: ", X[3], Y[3])
    print("transformed dataset: ", transformedDataset[3])
    print("original type: ", type(X[3]))
    print("transformed type: ", type(transformedDataset[3][0]))

    print("### If incorrect transformation applied on the numpy array ###")

    incorrectTransform = CustomDataset(
        X, Y, data_transform=tf.CenterCrop(10), target_transform=tf.CenterCrop(10)
    )

    try:
        print("Exception raised or not? Let's see ----> ",
              incorrectTransform[3])
    except Exception as E:
        print("Yes!! it works if it's a type error with our message ---> ", E)
