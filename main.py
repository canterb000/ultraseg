import torch
from model import unet
from torch.autograd import Variable
import numpy as np

def main():

    model = unet()


#TODO input constraint 48*
    test_x = Variable(torch.FloatTensor(np.random.random((1, 1, 48, 48))))
    out_x = model(test_x)
    print(out_x)

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
