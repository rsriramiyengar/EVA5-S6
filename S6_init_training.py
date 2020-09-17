from torch.optim.lr_scheduler import StepLR

def init_training(model, device, train_loader, epochs, step_lr=True, l1_lambda=None, l2_en=False):

    if l2_en:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=1e-5, nesterov=False)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if step_lr:
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(epochs):

        train(model, device, train_loader, optimizer, epoch, l1_lambda)

        if step_lr:
            scheduler.step()

        print('\n Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
        test(model, device, test_loader)