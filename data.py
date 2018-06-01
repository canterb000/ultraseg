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
                fmask = os.path.basename(fname)
                fori = fmask[:-9] + '.tif'

                self.dataset.append([os.path.join(rootdir, fori), os.path.join(rootdir, fmask), fori])
                    #print("ori: {}, mask: {}".format(os.path.join(rootdir, fori), os.path.join(rootdir, fmask) ))
                totalcount+=1

            print("train db totalcount: {}".format(totalcount))
            self.count = totalcount

        else:
            print("test database")
            totalcount = 0
            for i in range(5508):
                fmask = str(i+1)+".tif"
                fullpath = os.path.join(rootdir, fmask)
                self.dataset.append([fullpath, fmask])
                #print("{} {} {}".format(i+1, fullpath, fmask))
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
            #print("before {}  shape {}".format(before, img.shape))
     #       testimg = Image.fromarray(img[0])
     #       testimg.save("pred/"+ fname)

            img = torch.from_numpy(img).float()
            #print("train:final {}".format(img.size))
            

            gt = Image.open(gt_path)
            before = gt.size
            gt = gt.resize((self.nCol, self.nRow), Image.ANTIALIAS)
     #       print("gtresize: {}".format(gt.size))
            gt = np.array(gt)
            gt = gt[0:self.nRow, 0:self.nCol]
      #      print("gt[]: {}".format(gt.shape))
            gt = np.atleast_3d(gt).transpose(2, 0, 1)
            gt = gt / 255.0
            gt = torch.from_numpy(gt).float()
            #print("train:final {}".format(gt.size))

            
            return img, gt, fname

        else:
            img_path, fname = self.dataset[idx]
            #print("fname: {}".format(fname))
            img = Image.open(img_path)
            #img.save("pred/testori_"+fname)
            before = img.size
            imgdata = np.array(img.resize((self.nCol, self.nRow), Image.ANTIALIAS))
            #img.save("pred/testresize_"+fname)
            img = imgdata[0:self.nRow, 0:self.nCol]
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            #print("{} {} {}".format(img, type(img), img.shape))
            img = torch.from_numpy(img).float()
            #save_tif(np.array(img), "testresize_"+fname)

           #print("testfinal:{}".format(img.size))

            return img, fname 
