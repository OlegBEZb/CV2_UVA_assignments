from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict
from img_utils import rgb2tensor


# Helper function to quickly see the values of a list or dictionary of data
def printTensorList(data, detailed=False):
    if isinstance(data, dict):
        print('Dictionary Containing: ')
        print('{')
        for key, tensor in data.items():
            print('\t', key, end='')
            print(' with Tensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print('}')
    else:
        print('List Containing: ')
        print('[')
        for tensor in data:
            print('\tTensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print(']')


def is_png(fname):
    ext = os.path.splitext(fname)[1]
    return ext.lower() == '.png'


def make_dataset(dir):
    files = []
    dir = os.path.expanduser(dir)
    for fname in sorted(os.listdir(dir)):
        if is_png(fname):
            path = os.path.join(dir, fname)
            files.append(path)
    return files


class SwappedDatasetLoader(Dataset):

    def __init__(self, data_path, transform=None, resize=256):
        # Define your initializations and the transforms here. You can also
        # define your tensor transforms to normalize and resize your tensors.
        # As a rule of thumb, put anything that remains constant here.
        if not os.path.isdir(data_path):
            raise
        self.file_paths = make_dataset(data_path)

        self.group2path_mapping = defaultdict(list)
        for file_path in self.file_paths:
            self.group2path_mapping[file_path.split('/')[-1].split('_')[0]].append(file_path)
        corrupted_sets = {k: v for k, v in self.group2path_mapping.items() if len(v) != 4}
        assert len(corrupted_sets) == 0, f'some of the image sets are corrupted\n{corrupted_sets}'

        print(f'initialized with {len(self.file_paths)} files. first path is', self.file_paths[0])
        print(f'different groups: {len(self.group2path_mapping)}. '
              f'first group: {self.group2path_mapping[list(self.group2path_mapping.keys())[0]]}')
        self.group_names = list(self.group2path_mapping.keys())
        self.resize = resize

    def __len__(self):
        # Return the length of the datastructure that is your dataset
        return len(self.group2path_mapping)

    def __getitem__(self, index):
        # Write your data loading logic here. It is much more efficient to
        # return your data modalities as a dictionary rather than list. So you
        # can return something like the follows:
        #     image_dict = {'source': source,
        #                   'target': target,
        #                   'swap': swap,
        #                   'mask': mask}

        #     return image_dict, self.data_paths[index]
        group_name = self.group_names[index]
        group_paths = self.group2path_mapping[group_name]

        try:
            mask = cv2.imread([p for p in group_paths if '_mask_' in p][0])
            mask = np.where(mask > 0, 1, mask)

            source = cv2.imread([p for p in group_paths if '_fg_' in p][0])

            target = cv2.imread([p for p in group_paths if '_bg_' in p][0])

            swap = cv2.imread([p for p in group_paths if '_sw_' in p][0])
        except:
            print(f'was not able to find some files for {group_name}\nhaving the following paths:{group_paths}')

        # CV2 uses BGR by default
        images_dict = {'source': rgb2tensor(source),
                       'target': rgb2tensor(target),
                       'swap': rgb2tensor(swap),
                       'mask': rgb2tensor(mask),
                       }
        # print('images_dict\n', {k: v.shape for k,v in images_dict.items()})
        return images_dict


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    # It is always a good practice to have separate debug section for your
    # functions. Test if your dataloader is working here. This template creates
    # an instance of your dataloader and loads 20 instances from the dataset.
    # Fill in the missing part. This section is only run when the current file
    # is run and ignored when this file is imported.

    # This points to the root of the dataset
    data_root = '../../data_set/data_set/data'
    # This points to a file that contains the list of the filenames to be
    # loaded.
    # test_list = ''
    print('[+] Init dataloader')
    # Fill in your dataset initializations
    testSet = SwappedDatasetLoader(
        # test_list,
        data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(20):
        a = time.time()
        i, images = next(enu)
        b = time.time()
        # Uncomment to use a prettily printed version of a dict returned by the
        # dataloader.
        printTensorList(images, True)
        print('[*] Time taken: ', b - a)
