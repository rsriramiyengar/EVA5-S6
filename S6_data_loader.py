def initialize_train_test_loader(batch_size = 128):

    print("\n Initialize train and test loader with Batch Size:{}".format(batch_size))

    torch.manual_seed(22)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                            transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
                            transforms.RandomRotation((-7.0,7.0), fill=(1,)),                          
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=batch_size, shuffle=True, **kwargs)
       
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader