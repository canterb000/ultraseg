import torch
import torch.utils.data as data
import glob
import os
import numpy as np
from scipy.ndimage import imread
from PIL import Image

def save_tif(ori, name):
    img = ori[0,:,:]
    img = img > 0.5
    img = Image.fromarray(np.uint8(img*255), mode='L')
    filename = name
    img.save('pred/'+filename) 
    print("Saved: {}".format(filename))



class datasetbuilder(data.Dataset):
    def __init__(self, rootdir, train, nRow, nCol):
        self.dataset = []
        self.count = 0
        self.train = train
        self.nRow = nRow
        self.nCol = nCol

        #print("dir:{} train:{} dim: {} * {}".format(rootdir, train, nRow, nCol))

        if train:
            totalcount = 0
            for fname in glob.glob(os.path.join(rootdir, '*_mask.tif')):
            #for fname in glob.glob(os.path.join(rootdir, '*_mask.png')):
                fmask = os.path.basename(fname)
                fori = fmask[:-9] + '.tif'
                #fori = fmask[:-9] + '.png'

                self.dataset.append([os.path.join(rootdir, fori), os.path.join(rootdir, fmask), fori])
                totalcount+=1

            print("train db totalcount: {}".format(totalcount))
            self.count = totalcount

        else:
            print("test database")
            totalcount = 0
            for i in range(5508):
                fmask = str(i+1)+".tif"
                #fmask = str(i+1)+".png"
                fullpath = os.path.join(rootdir, fmask)
                self.dataset.append([fullpath, fmask])
                totalcount += 1
            self.count = totalcount

        print("count: {}".format(self.count))


    def __len__(self):
        return self.count


    def __getitem__(self, idx):
        if self.train:
            img_path, gt_path, fname = self.dataset[idx]
            img = Image.open(img_path)
            before = img.size 

            img = img.resize((self.nCol, self.nRow), Image.ANTIALIAS)

            imgdata = np.array(img)

            img = imgdata[0:self.nRow, 0:self.nCol]

            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img).float()
            

            gt = Image.open(gt_path)
            before = gt.size
            gt = gt.resize((self.nCol, self.nRow), Image.ANTIALIAS)
            gt = np.array(gt)
            gt = gt[0:self.nRow, 0:self.nCol]
            gt = np.atleast_3d(gt).transpose(2, 0, 1)
            gt = gt / 255.0
            gt = torch.from_numpy(gt).float()

            
            return img, gt, fname

        else:
            img_path, fname = self.dataset[idx]
            img = Image.open(img_path)
            before = img.size
            imgdata = np.array(img.resize((self.nCol, self.nRow), Image.ANTIALIAS))
            img = imgdata[0:self.nRow, 0:self.nCol]
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img).float()


            return img, fname 
