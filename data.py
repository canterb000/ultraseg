import torch
import torch.utils.data as data
import glob
import os
import numpy as np
from scipy.ndimage import imread

class datasetbuilder(data.Dataset):
    def __init__(self, rootdir, train, nRow, nCol):
        self.dataset = []
        self.count = 0
        self.train = train
        self.nRow = nRow
        self.nCol = nCol

        print("dir:{} train:{} dim: {} * {}".format(rootdir, train, nRow, nCol))

        if train:
            totalcount = 0
            for fname in glob.glob(os.path.join(rootdir, '*_mask.tif')):
                fmask = os.path.basename(fname)
                fori = fmask[:-9] + '.tif'

                self.dataset.append([os.path.join(rootdir, fori), os.path.join(rootdir, fmask), fname])
                if totalcount < 3:
                    print("ori: {}, mask: {}".format(os.path.join(rootdir, fori), os.path.join(rootdir, fmask) ))
                totalcount+=1

            print("train db totalcount: {}".format(totalcount))
            self.count = totalcount

        else:
            print("test database")
            totalcount = 0
            for fname in glob.glob(os.path.join(rootdir, '*.tif')):
                self.dataset.append(os.path.join(rootdir, fname))
#                print("test: {}".format(os.path.join(rootdir, fname)))
                totalcount+=1
            print("test db totalcount: {}".format(totalcount))
            self.count = totalcount


        print("count: {}".format(self.count))


    def __len__(self):
        return self.count


    def __getitem__(self, idx):
        if self.train:
            img_path, gt_path, fname = self.dataset[idx]
            img = imread(img_path)

            img = img[0:self.nRow, 0:self.nCol]
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img).float()

            gt = imread(gt_path)[0:self.nRow, 0:self.nCol]
            gt = np.atleast_3d(gt).transpose(2, 0, 1)
            gt = gt / 255.0
            gt = torch.from_numpy(gt).float()

            return img, gt, fname

        else:
            img_path = self.dataset[idx]
            img = imread(img_path)

            img = img[0:self.nRow, 0:self.nCol]
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img).float()

            return img
