import torch
class Args:
    #share
    test_interval = 50
    batch_size = 256
    total_epochs = 20
    lr = 1e-3
    weight_decay = 1e-4
    # neural network
    dropout = 0.5
    # logs
    folder = 'resnet-9'
    save_model = True
    save_interval = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

