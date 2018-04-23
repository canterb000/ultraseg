import torch
from model import unet
from data import datasetbuilder
from torch.autograd import Variable
import numpy as np
import argparse
import os
from PIL import Image

def main():

    model = unet()

    dim = [400,560]
#TODO input constraint 48*
    test_x = Variable(torch.FloatTensor(np.random.random((1, 1, 48, 48))))
    out_x = model(test_x)
    print(out_x)

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='data dir', default='/home/ecg/Downloads/segdata')
    parser.add_argument('--batchsize', type=int, help='batch size', default='1')
    parser.add_argument('--workersize', type=int, help='worker number', default='2')
    args = parser.parse_args()
    print(args)
   

    traindata = datasetbuilder(rootdir=os.path.join(args.datadir, 'train'), train=True, nRow=dim[0], nCol=dim[1]) 
    #testdata = datasetbuilder(os.path.join(args.datadir, 'test'), train=False, dim[0], dim[1]) 
     
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batchsize,
                                             num_workers=args.workersize, shuffle=True)


    for i, (x, y, name) in enumerate(train_loader):
        if i == 0:
            print("name:{}".format(name))
            img = x[0,0,:,:]
            img = img > 0.5
            img = Image.fromarray(np.uint8(img*255), mode='L')
            img.save("train-ground.tif")

            img2 = y[0,0,:,:]
            img2 = img2 > 0.5
            img2 = Image.fromarray(np.uint8(img2*255), mode='L')
            img2.save("train-mask.tif")
            break

'''    
    params = 
    args = 
    optimizer = 
    data_loader = 
    loss = 


    for : 
        train()
        save()
    for :
        test()

''' 

if __name__ == '__main__':
    main()
