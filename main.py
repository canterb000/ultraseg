import wfdb
import sys
import torch
from model import unet
from data import datasetbuilder
from torch.autograd import Variable
import numpy as np
import argparse
import os
from PIL import Image, ImageDraw
import torch.optim as optim
import torch.nn as nn
from findtools.find_files import (find_files, Match)
import ntpath
import matplotlib.pyplot as plt


PIXEL_COUNT_TH = 300
def main():
    dim = [480,640]
#TODO input constraint 48*
    #test_x = Variable(torch.FloatTensor(np.random.random((1, 1, 48, 48))))
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--datadir', type=str, help='data dir', default='/home/ecg/Downloads/segdata')
    parser.add_argument('--datadir', type=str, help='data dir', default='/home/ecg/Public/ultraseg/ultraseg/ecgdata')
    parser.add_argument('--batchsize', type=int, help='batch size', default='1')
    parser.add_argument('--workersize', type=int, help='worker number', default='1')
    parser.add_argument('--cuda', help='cuda configuration', default=True)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--epoch', type=int, help='epoch', default=5)
    parser.add_argument('--checkpoint', type=str, help='output checkpoint filename', default='checkpoint.tar')
    parser.add_argument('--resume', type=str, help='resume configuration', default='checkpoint.tar')
    parser.add_argument('--start_epoch', type=int, help='init value of epoch', default='0')
    parser.add_argument('--output_csv', type=str, help='init value of epoch', default='output.csv')
    
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
        model = model.cuda(1)

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
        lossfn = lossfn.cuda(1)
    loss_sum = 0


    print("######Train:#######")
    for epoch in range(args.start_epoch, args.epoch):
        print("rangetest: epoch: {}".format(epoch))
        for i, (x, y, name) in enumerate(train_loader):
            x, y = Variable(x), Variable(y)
            if args.cuda:
                x = x.cuda(1)
                y = y.cuda(1)
 
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



    #Detection Process
    originfiledir="/home/ecg/Public/ecgdatabase/mitdb"

    HALF_OFFSET = 300
    FREQ = 360
    txt_files_pattern = Match(filetype = 'f', name = '*.dat')
    found_files = find_files(path=originfiledir, match=txt_files_pattern)
    

    ###Preprocessing
    for found_file in found_files:
        head, tail = ntpath.split(found_file)
        recordname = tail.split('.')[0]
        readdir = head + '/' + recordname
        print("{}".format(readdir)) 
        sampfrom = 0
        sampto = sampfrom + 2 * HALF_OFFSET
        record = wfdb.rdsamp(readdir,  sampfrom = sampfrom)
        recordlength = len(record.p_signals)
        while sampto < recordlength:
            record = wfdb.rdsamp(readdir, sampfrom = sampfrom, sampto = sampto)
            
            sampfrom +=  HALF_OFFSET
            sampto +=  HALF_OFFSET
            
            #####detect qrs. and R-peak loc and drop R if qrs is in the next window
            p_signal = record.p_signals[:, 0]
            freq = record.fs
            x = np.linspace(0,  HALF_OFFSET * 2, HALF_OFFSET * 2)
            plt.plot(x, p_signal)
            plt.axis('off')
            plt.ylim(-2, 2.5)
            signalpath = 'snapshot.png'
            plt.savefig(signalpath)

            img = Image.open(signalpath).convert('L')

            img = img.resize((dim[1], dim[0]), Image.ANTIALIAS)
            imgdata = np.array(img)
            img = imgdata[0:dim[0], 0:dim[1]]
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())

            img = np.expand_dims(img, axis=0)

            img = torch.from_numpy(img).float()
            x = img.cuda(1)
            
            print("img: {}, \n x:{}".format(img, x))
            y = model(Variable(x))
            y = y.cpu().data.numpy()[0,0]
            labelflag = str(x) 
            res, start, end = qrs_classify(y, labelflag)
            img = img[0,0]
            img = img > 0.5
            img = np.array(img)
            print("img: {} shape {} ".format(img, img.shape))
            h, w = img.shape
            start = -1
            end = -1
            trailcount = 5
            flag = False
            for wi in range(w):
                pixelsum = 0
                for hi in range(h):
                    val = img[hi, wi]
                    pixelsum += val
                    if pixelsum > PIXEL_COUNT_TH:
                        break
                if pixelsum < 50:
                    trailcount -= 1
                    if trailcount < 0: 
                        trailcount = 5
                        print("{} - {}".format(start, end))
                        start = end
                        end = -1
                        flag = False
                if flag and start > 0 and pixelsum > PIXEL_COUNT_TH:
                    end = wi
                    trailcount = 5
                if (not flag) and pixelsum > PIXEL_COUNT_TH:
                    start = wi
                    flag = True
       
            
    
            


            print("res: {}".format(res))
            save_tif(y, x.cpu().numpy()[0,0], str(sampfrom), labelflag, start, end)
 
            sys.exit()
            #####locate the qrs width and output qrs png. later for classification.  store in the seires.  
            #####calculate heart rate; heart rate anomaly detection
            #####
          
            
           
    ###############################   
    print("######QuickTest:#######")
    acc = 0
    samplecount = 0
    for i, (dat, name, label) in enumerate(test_loader):
        if '1' in label:
            labelflag = True
            #print("label check: {}, {}".format(label, labelflag))
        elif '0' in label:
            labelflag = False
            #print("label check: {}, {}".format(label, labelflag))
        x = dat.cuda(1)
        print("dat {}, \n x {}".format(dat, x))

        #if torch.cuda.is_available():
        y = model(Variable(x))
        y = y.cpu().data.numpy()[0,0]
        res, start, end = qrs_classify(y, labelflag)
        filename = name[0][:-4]
        if res:
            acc += res 
            save_tif(y, x.cpu().numpy()[0,0], filename, labelflag, start, end)
        else:
            print("miss: {}, {}".format(res, name[0])) 
            save_tif(y, x.cpu().numpy()[0,0], filename, labelflag, start, end)
        samplecount = i+1
        #save_tif(ori.cpu().numpy()[0,0], name[0])

    print("count: {} acc: {}".format(samplecount, acc/samplecount))
'''
    for i, (x, y, name) in enumerate(train_loader):
        if i > 5630:
            x = x.cuda()
            y_pred = model(Variable(x))
            y_pred = y_pred.cpu()
            x = x.cpu()
            save_tif(y_pred.data.numpy(), name='pred_'+str(i))
            save_tif(y.numpy(), name='gt_'+str(i))
            tif_to_tensor(str(i), y)
        else:
            continue
'''
def qrs_classify(ori, truth):
    img = ori[:,:]
    img = img > 0.5
    img = np.array(img)
    h, w = img.shape
    start = -1
    end = -1
    flag = False
    for wi in range(w):
        pixelsum = 0
        for hi in range(h):
            val = img[hi, wi]
            pixelsum += val
            if pixelsum > PIXEL_COUNT_TH:
                break
        if flag and start > 0 and pixelsum > PIXEL_COUNT_TH:
            end = wi
        if (not flag) and pixelsum > PIXEL_COUNT_TH:
            start = wi
            flag = True
        
    if truth == flag:
        return 1, start, end
    else:
        print("flag: {} , truth: {}".format(flag, truth))
        return 0, start, end


def save_tif(pred, ori, name, labelflag, start, end):
    img = ori[:,:]
    img_y = pred[:,:]
    img = img > 0.5
    img_y = img_y > 0.5
    img = Image.fromarray(np.uint8(img*255), mode='L')
    img_y = Image.fromarray(np.uint8(img_y*255), mode='L')
    if start > 0 and end > 0:
        img_edit = ImageDraw.Draw(img)
        img_edit.line([(start, 0),(start, 420)], "black")
        img_edit.line([(end, 0),(end, 420)], "black")

    filename = name
    label = '0'
    if labelflag:
        label = '1'
    img.save('testoutput/' + label + '_' + filename + '_' + str(start) + '_' + str(end) + '.png') 
    img_y.save('testoutput/' + label + '_' + filename + '_mask_' + str(start) + '_' + str(end) + '.png') 
    print("Saved: {}".format(filename))


def save_checkpoint(state, filename):
    torch.save(state, filename)

def tif_to_tensor(output, name, tif):
    f = open(output, 'a+')
    f.write(name + ',')
   
    imgdata= (tif.data)[0,0]
    img = imgdata > 0.5
    img = Image.fromarray(np.uint8(img*255), mode='L')
    img = img.resize((580, 420), Image.ANTIALIAS)
    img.save("pred/"+ name + ".tif")
    img = np.array(img)
    
    h, w = img.shape
    flag=False
    start=-1
    length=-1

    #print("h: {} , w: {} ".format(h, w))

    for wi in range(w):
        for hi in range(h):
            val = img[hi, wi]
            #pos = hi + wi*420 
            pos = hi + wi*420 + 1
            if val > 0.5:
                if flag:
                    length+=1
                else:
                    start=pos
                    length=1
                    flag=True
                
                #print("{} {}   binary: {} {}".format(hi, wi, flag, length))
            else:
                if flag:
                    #print("{},{}".format(start, length))
                    #if int(name) < 1:
                    f.write("{} {} ".format(start, length))
                    #f.write("{},{}---{}\t".format(hi, wi, length))
                    
                    start=-1
                    length=-1
                    flag=False
    f.write("\n")
    f.close() 



if __name__ == '__main__':
    main()
