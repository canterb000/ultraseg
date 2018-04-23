from model import unet

def main():

    model = unet()

    test_x = Variable(torch.FloatTensor(1, 3, 1024, 1024))
    out_x = net(test_x)

    print(out_x.size())
    
#    print("model: {}".format(model))
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
