import torch
from model import unet
from data import datasetbuilder
from torch.autograd import Variable
import numpy as np
import argparse
import os
from PIL import Image
import torch.optim as optim
import torch.nn as nn

def main():

    dim = [400,560]
#TODO input constraint 48*
    test_x = Variable(torch.FloatTensor(np.random.random((1, 1, 48, 48))))

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='data dir', default='/home/ecg/Downloads/segdata')
    parser.add_argument('--batchsize', type=int, help='batch size', default='1')
    parser.add_argument('--workersize', type=int, help='worker number', default='2')
    parser.add_argument('--cuda', help='cuda configuration', default=True)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
    parser.add_argument('--epoch', type=int, help='epoch', default=40)
    parser.add_argument('--checkpoint', type=str, help='output checkpoint filename', default='checkpoint.tar')
    parser.add_argument('--resume', type=str, help='resume configuration', default='checkpoint.tar')
    parser.add_argument('--start_epoch', type=int, help='init value of epoch', default='0')
    
    args = parser.parse_args()
    print(args)

    traindata = datasetbuilder(rootdir=os.path.join(args.datadir, 'train'), train=True, nRow=dim[0], nCol=dim[1]) 
    testdata = datasetbuilder(rootdir=os.path.join(args.datadir, 'test'), train=False, nRow=dim[0], nCol=dim[1])
     
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batchsize,
                                             num_workers=args.workersize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=args.batchsize,
                                             num_workers=args.workersize, shuffle=False)

    model = unet()
    if args.cuda:
         model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {}, loss {})".format(checkpoint['epoch'], checkpoint['loss']) )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    lossfn = nn.MSELoss()
    if args.cuda:
        lossfn = lossfn.cuda()
    loss_sum = 0


    print("######Train:#######")
    train_size = 5300
    for epoch in range(args.start_epoch, args.epoch):
        print("rangetest: epoch: {}".format(epoch))
        for i, (x, y, name) in enumerate(train_loader):
            if i > train_size:
                break
            x, y = Variable(x), Variable(y)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
 
            y_pred = model(x)

            loss = lossfn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            loss_sum += loss.data[0]
            optimizer.step()

            if i % 100 == 0:
                print('Iter: {}, Loss: {}'.format(i, loss.data[0]))

        print('Epoch: {}, Epoch Loss: {}'.format(epoch, loss.data[0] / len(train_loader)))

        save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'loss': loss.data[0] / len(train_loader)
        }, args.checkpoint)


        
    print("######QuickTest:#######")
    for i, (x, y, name) in enumerate(train_loader):
        if i > train_size:
            x = x.cuda()
            y_pred = model(Variable(x))
            y_pred = y_pred.cpu()
            x = x.cpu()
            save_tif(x.numpy(), name='ori_'+str(i))
            save_tif(y_pred.data.numpy(), name='pred_'+str(i))
            save_tif(y.numpy(), name='gt_'+str(i))
        else:
            break


def save_tif(ori, name):
    img = ori[0,0,:,:]
    img = img > 0.5
    img = Image.fromarray(np.uint8(img*255), mode='L')
    filename = name + '.tif'
    img.save('pred/'+filename) 
    print("Saved: {}".format(filename))


def save_checkpoint(state, filename):
    torch.save(state, filename)



if __name__ == '__main__':
    main()
