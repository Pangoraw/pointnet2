"""
    Face dataset
"""

import os
import os.path
import h5py
import numpy as np


class FaceDataset():
    def get_data_file(self):
        file = f"{self.split}_hdf5_file_list.txt"
        with open(file, "r") as f:
            files = [ls.rstrip() for ls in f]
        return files

    def __init__(self, root, npoints=2048, classification=False, split='train', normalize=True, return_cls_label=False):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.catfile = os.path.join(self.root, 'all_object_categories.txt')
        self.cat = {}

        self.classification = classification
        self.return_cls_label = return_cls_label

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        data_files = self.get_data_file()
        samples = np.array([], dtype='float32')
        labels = np.array([], dtype='uint8')
        segs = np.array([], dtype='uint8')
        for file in data_files:
            store = h5py.File(file, 'r')
            samples = np.concatenate(samples, store['data'])
            labels = np.concatenate(labels, store['label'])
            segs = np.concatenate(segs, store['pid'])

        self.samples = samples
        self.labels = labels
        self.segs = segs
        self.seg_classes = np.unique(segs.reshape(-1,))


    def __getitem__(self, index):
        if self.classification:
            return self.samples[index], self.labels[index]
        else:
            return self.samples[index], self.segs[index]


    def __len__(self):
        return self.samples.shape[0]
