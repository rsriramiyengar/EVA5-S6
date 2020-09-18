from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from S6.S6_train_test_function import train
from S6.S6_train_test_function import test
from S6.S6_data_loader import init_train_test_loader
from torch.optim.lr_scheduler import StepLR


train_loader, test_loader = init_train_test_loader()

def init_training(model, device, train_loader, epochs, step_lr=True, l1_lambda=None, l2_en=False):

    if l2_en:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=1e-5, nesterov=False)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if step_lr:
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    train_losses=[]
    train_acc=[]
    test_losses=[]
    test_acc=[]
    for epoch in range(epochs):

        train_losses1,train_acc1=train(model, device, train_loader, optimizer, epoch, l1_lambda)
        train_losses.append(train_losses1)
        train_acc.append(train_acc1)
        if step_lr:
            scheduler.step()

        print('\n Epoch {}, lr {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        test_losses1,test_acc1=test(model, device, test_loader)
        test_losses.append(test_losses1)
        test_acc.append(test_acc1)
        
    return train_losses,train_acc,test_losses,test_acc
    
