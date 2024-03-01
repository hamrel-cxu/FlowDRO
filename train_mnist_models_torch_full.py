import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import utils_flowdro as futils
import scipy

### Deprecated see bottom

def get_data_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def infinite_data_loader(data_loader):
    while True:
        for data in data_loader:
            yield data

class CNN(nn.Module):
    def __init__(self, img_rows=28, img_cols=28, channels=1, nb_filters=64, nb_classes=10):
        super(CNN, self).__init__()
        self.activation = nn.ELU()
        self.conv1 = nn.Conv2d(channels, nb_filters, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2, kernel_size=6, stride=2, padding=0)
        self.conv3 = nn.Conv2d(nb_filters * 2, nb_filters * 2, kernel_size=5, stride=1, padding=0)
        self.fc = nn.Linear(self._calc_input_feats(img_rows, img_cols, nb_filters), nb_classes)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _calc_input_feats(self, img_rows, img_cols, nb_filters):
        size = (img_rows, img_cols)  # Initial size
        size = (size[0] - 8) // 2 + 1, (size[1] - 8) // 2 + 1  # After conv1
        size = (size[0] - 6) // 2 + 1, (size[1] - 6) // 2 + 1  # After conv2
        size = (size[0] - 5) // 1 + 1, (size[1] - 5) // 1 + 1  # After conv3
        return size[0] * size[1] * nb_filters * 2

def frm_wrapper(store_loss = True):    
    for _ in range(FRM_steps):
        # Sample batch_size number of samples in num_batches
        rand_idx = torch.randperm(full_X.size(0))[:batch_size]
        data, target = full_X[rand_idx], full_y[rand_idx]
        data, target = data.to(device), target.to(device)
        loss_ls = frm(x=data, y=target, store_loss=store_loss)
        for c in range(num_classes):
            optimizer_flow_ls[c].zero_grad()
            loss_ls[c].backward()
            optimizer_flow_ls[c].step()
            scheduler_ls[c].step()       
    with torch.no_grad():
        # Use for inference, by aligning the data
        target_ls = []
        perturbed_data_ls = []
        for c in range(num_classes):
            idx_c = target ==c
            target_ls.append(target[idx_c])
            perturbed_data_ls.append(flow_model_ls[c](data[idx_c])[-1])
        target = torch.cat(target_ls, dim=0)
        perturbed_data = torch.cat(perturbed_data_ls, dim=0)
    return perturbed_data, target

def frm(x, y, store_loss = True):
    def get_loss1_and_2(predz, x, y):
        loss1 = criterion(model(predz[-1]), y)
        diff = (predz[-1]-x).view(x.size(0), -1)
        loss2 = 0.5/gamma*torch.norm(diff, 2, 1).pow(2).mean()
        num_x = x.size(0)
        return num_x*loss1, num_x*loss2, loss1-loss2
    criterion = nn.CrossEntropyLoss()
    loss1_1_ls, loss2_1_ls, loss_ls = [], [], []
    for c in range(num_classes):
        idx = y==c
        if idx.sum() == 0:
            # No sample in this class
            loss1_1_ls.append(0)
            loss2_1_ls.append(0)
            loss_ls.append(0)
            continue
        predz = flow_model_ls[c](x[idx])
        loss1, loss2, loss = get_loss1_and_2(predz, x[idx], y[idx])
        loss1_1_ls.append(loss1)
        loss2_1_ls.append(loss2)
        loss_ls.append(loss)
    num_x_tot = x.size(0)
    loss1, loss2 = sum(loss1_1_ls)/num_x_tot, sum(loss2_1_ls)/num_x_tot
    if args.mtd == 'FRM' and store_loss:
        args.loss_LFD_classifier += [loss1.item()]
        args.loss_LFD_w2 += [-loss2.item()]
    return loss_ls
    
def wrm(x, y, store_loss = True):
    x_adv = x.clone()
    x = x.detach()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_classifier_ls, loss_w2_ls = [], []
    num_sample = x.size(0)
    for t in range(WRM_steps):
        x_adv.requires_grad_()
        loss = criterion(model(x_adv), y)
        diff = (x_adv-x).view(x.size(0), -1)
        loss_2 = 0.5/gamma*torch.norm(diff, 2, 1).pow(2).sum()
        loss_full = loss - loss_2
        grad = torch.autograd.grad(loss_full, x_adv)[0]
        alpha_t = (1.0 / np.sqrt(t + 2))
        x_adv = (x_adv + alpha_t * grad).detach()
        loss_classifier_ls.append(loss.item()/num_sample)
        loss_w2_ls.append(-loss_2.item()/num_sample)
    if args.mtd == 'WRM' and store_loss:
        args.loss_LFD_classifier += loss_classifier_ls
        args.loss_LFD_w2 += loss_w2_ls
    return x_adv

def train_frm(model, device, optimizer):
    criterion = nn.CrossEntropyLoss()
    # Train FRM attacker
    perturbed_data, target = frm_wrapper(store_loss=True)
    perturbed_output = model(perturbed_data)
    # Forward pass with adversarial examples
    loss_adv = criterion(perturbed_output, target)
    optimizer.zero_grad()
    loss_adv.backward()
    args.loss_classifier.append(loss_adv.item())
    optimizer.step()
    scheduler.step()

def train_wrm(model, device, optimizer):
    criterion = nn.CrossEntropyLoss()
    # Generate adversarial examples
    data, target = next(train_loader)
    data, target = data.to(device), target.to(device)
    perturbed_data = wrm(x=data, y=target)
    perturbed_output = model(perturbed_data)
    # Forward pass with adversarial examples
    loss_adv = criterion(perturbed_output, target)
    optimizer.zero_grad()
    loss_adv.backward()
    args.loss_classifier.append(loss_adv.item())
    optimizer.step()
    scheduler.step()
    

# (5) Evaluation Script
def model_eval(model, loader):
    true = 0
    tot = 0 
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        tot += target.size(0)
        true += (predicted == target).sum().item()
    return true / tot

def evaluate():
    save_model(f'_step{FRM_steps}_run{args.run}')

def loss_convolve(loss_ls, window_size = 25):
    if len(loss_ls) < window_size:
        return loss_ls
    else:
        return scipy.signal.convolve(loss_ls, np.ones(window_size)/window_size, 
                            mode='valid', method = 'fft')

def save_model(suffix = ''):
    state_dict = {
        'batch': batch+1,
        'state_dict': model.state_dict(),
        'loss_classifier': args.loss_classifier,
        'loss_LFD_classifier': args.loss_LFD_classifier,
        'loss_LFD_w2': args.loss_LFD_w2
    }
    filename = filename_raw.split('.')[0] + f'{suffix}.h5'
    torch.save(state_dict, filename)

parser = argparse.ArgumentParser()
parser.add_argument('--mtd', type=str, default='FRM', choices=['WRM', 'FRM'])
parser.add_argument('--run', type=int, default=0)
args = parser.parse_args()
gamma = 5
FRM_steps = 3
WRM_steps = 100
if __name__ == '__main__':
    import importlib
    importlib.reload(futils)
    batch_size = 512
    batch_tot = 11000 # Each epoch has abot 118 batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # (1) Data Loading
    train_loader, test_loader = get_data_loader(batch_size)
    full_data = [batch for batch in train_loader]
    train_loader = infinite_data_loader(train_loader)
    full_X = torch.cat([batch[0] for batch in full_data], dim=0).to(device)
    full_y = torch.cat([batch[1] for batch in full_data], dim=0).to(device)
    # (2) Model and training
    train_func_dict = {'WRM': train_wrm, 'FRM': train_frm}
    model = CNN().to(device)
    num_classes = 10
    flow_model_ls = [futils.get_flowmodel() for _ in range(num_classes)]
    lr_flow = 1e-4
    optimizer_flow_ls = [optim.Adam(flow_model.parameters(), lr=lr_flow, maximize=True) for flow_model in flow_model_ls]
    scheduler_ls = [optim.lr_scheduler.StepLR(optimizer_flow, step_size=1, gamma=1) for optimizer_flow in optimizer_flow_ls]
    print(flow_model_ls[0])
    #####
    lr_cnn = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr_cnn)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    train_func = train_func_dict[args.mtd]
    prefix = args.mtd.lower()
    dir_name = os.path.join('models', 'mnist_full')
    os.makedirs(dir_name, exist_ok=True)
    filename_raw = os.path.join(dir_name, f'{prefix}.h5')
    start_batch = 1
    args.loss_classifier, args.loss_LFD_classifier, args.loss_LFD_w2 = [], [], []
    for batch in tqdm(range(start_batch, batch_tot + 1)):
        train_func(model, device, optimizer)
        if batch % 100 == 0:
            print(f'After batch {batch}/{batch_tot}, loss = {loss_convolve(args.loss_classifier)[-1]:.2e}')
        if batch % 200 == 0 or batch == batch_tot:
            evaluate()    