"""
    Face dataset
"""

import os
import os.path
import h5py
import numpy as np


class FaceDataset():
    def get_data_file(self):
        file = os.path.join(self.root, "{}_hdf5_file_list.txt".format(self.split))
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
        file_path = os.path.join(self.root, data_files[0])
        store = h5py.File(file_path, 'r')
        samples = store['data'].value
        labels = store['label'].value
        segs = store['pid'].value

        self.samples = samples
        self.labels = labels
        self.segs = segs
        classes = np.unique(segs.reshape(-1, ))
        self.seg_classes = {0: classes}
        self.n_classes = classes.shape[0]
        self.classes = {0: 'face'}

    def __getitem__(self, index):
        if self.classification:
            return self.samples[index], self.labels[index]
        elif self.return_cls_label:
            return self.samples[index], self.segs[index], self.labels[index]
        else:
            return self.samples[index], self.segs[index]

    def __len__(self):
        return self.samples.shape[0]
