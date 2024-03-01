import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset, Dataset
import os
import argparse
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomMNISTDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        data, label = self.mnist_dataset[index]
        if label == two_digits[0]:
            label = 0
        elif label == two_digits[1]:
            label = 1
        else:
            raise ValueError(f'Label {label} not in {two_digits}')
        return data, label

def get_data_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in two_digits]
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in two_digits]
    train_dataset = CustomMNISTDataset(Subset(train_dataset, train_indices))
    test_dataset = CustomMNISTDataset(Subset(test_dataset, test_indices))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_data_loader_full(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# (2) CNN Model
class CNN(nn.Module):
    def __init__(self, img_rows=28, img_cols=28, channels=1, nb_filters=64, nb_classes=2):
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

def normalize_tensor(tensor_now):
    reshape_tensor = tensor_now.view(tensor_now.size(0), -1)
    norm_tensor = torch.clip(torch.norm(reshape_tensor, p=2, dim=1), min=1e-10)
    normalized_tensor = reshape_tensor / norm_tensor.view(-1, 1)
    normalized_tensor = normalized_tensor.view(tensor_now.size())
    return normalized_tensor

def pgd(x, y, model, steps=15, p=2, epsilon=1):
    # Initialize adversarial example with the original input
    x_adv = x.clone()
    x.detach()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    for t in range(steps):
        x_adv.requires_grad_()
        loss = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        if p == 2:
            # Normalize the gradient for L2 norm
            step = epsilon*normalize_tensor(grad)
        elif p == np.inf:
            # Take the sign of the gradient for Linf norm
            step = epsilon*torch.sign(grad)
        else:
            raise ValueError("Unsupported norm type. Use 2 or np.inf")        
        # Apply the update
        alpha_t = (1.0 / np.sqrt(t + 2))
        x_adv = (x_adv + alpha_t * step).detach()
        # Project back into the epsilon-ball around the original image
        if p == 2:
            # For L2 norm
            perturbation = x_adv - x
            scaled_perturbation = epsilon*normalize_tensor(perturbation)
            x_adv = x + scaled_perturbation
        elif p == np.inf:
            # For Linf norm
            x_adv = torch.clip(x_adv, x - epsilon, x + epsilon)
    return x_adv

def get_attack_loader(model, loader, p=2, epsilon=1, attack_mtd='PGD'):
    new_loader = []
    norms_perturbed = []
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        # Generate adversarial examples
        if attack_mtd == 'PGD':
            perturbed_data = pgd(x=data, y=target, model=model, p=p, epsilon=epsilon)
        elif attack_mtd == 'WN':
            raise NotImplementedError('White noise attack not implemented')
        elif attack_mtd == 'Flow':
            raise NotImplementedError('Flow attack not implemented')
        else:
            raise ValueError('Unsupported attack method')
        new_loader.append((perturbed_data, target))
        diff = (perturbed_data-data).view(len(data), -1)
        if p == 2:
            norms_perturbed.append(torch.norm(diff, p=2, dim=1))
        else:
            norms_perturbed.append(torch.norm(diff, p=float('inf'), dim=1))
    norm_perturbed = torch.cat(norms_perturbed, dim=0).mean()
    frac = frac2 if p == 2 else fracinf
    print(f'Attack with {attack_mtd} mtd at p={p} & fraction = {frac}')
    if torch.isnan(norm_perturbed):
        raise ValueError('NaN encountered')
    print(f'Emprical movement: {norm_perturbed:.4f}')
    print(f'Intended movement: {epsilon:.4f}')
    return new_loader

def model_eval(model, loader):
    model.eval()
    true = 0
    tot = 0 
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        tot += target.size(0)
        true += (predicted == target).sum().item()
    return true / tot

def evaluate(epsilon = 0, p = 2, attack_mtd = 'PGD'):
    # Accuracy of the model on PGD adversarial examples
    if epsilon == 0:
        acc_attack = model_eval(model, test_loader)
    else:
        attack_loader = get_attack_loader(model, test_loader, epsilon=epsilon, p=p, attack_mtd=attack_mtd)
        acc_attack = model_eval(model, attack_loader)
    print(f'Test accuracy on {attack_mtd} attacked examples: {acc_attack:.4f}')
    print('##########################')
    return 1-acc_attack

parser = argparse.ArgumentParser()
parser.add_argument('--mtd', type=str, default='FRM', choices=['ERM', 'WRM', 'FRM'])
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--full', type=int, default=0, choices=[0, 1])
args = parser.parse_args()
FRM_steps = 3
attack_mtd = 'PGD'
two_digits = [int(d) for d in '6-7'.split('-')]
if __name__ == '__main__':
    suffix = f'_step{FRM_steps}_run{args.run}'
    nb_classes = 10 if args.full == 1 else 2
    if args.full == 1:
        dir_name = os.path.join('models', f'mnist_full')
        dir_name2 = os.path.join('results', f'mnist_full')
    else:
        dir_name = os.path.join('models', f'mnist_binary')
        dir_name2 = os.path.join('results', f'mnist_binary')
    os.makedirs(dir_name2, exist_ok=True)
    # Constants
    batch_size = 512
    # (1) Data Loading
    if args.full == 1:
        train_loader, test_loader = get_data_loader_full(batch_size)
    else:
        train_loader, test_loader = get_data_loader(batch_size)
    # (2) Model and training
    model = CNN(nb_classes=nb_classes).to(device)
    prefix = args.mtd.lower()
    filename = os.path.join(dir_name, f'{prefix}{suffix}.h5')
    if os.path.exists(filename) is False:
        raise ValueError('Model not found')
    print('Resume training from checkpoint')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    file_risks = {'ell_2': [], 'ell_inf': []}
    fracs_ell2 = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    fracs_ellinf = [0, 0.1, 0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]
    C2, Cinf = 10.35, 1
    if args.full == 1:
        C2 = 9.21
    for frac2, fracinf in zip(fracs_ell2, fracs_ellinf):
        epsilon2, epsiloninf = frac2*C2, fracinf*Cinf
        err_l2 = evaluate(epsilon2, 2, attack_mtd)
        file_risks['ell_2'].append(err_l2)
        err_linf = evaluate(epsiloninf, np.inf, attack_mtd)
        file_risks['ell_inf'].append(err_linf)
    file_risks['ell_2'] = np.array(file_risks['ell_2'])
    file_risks['ell_inf'] = np.array(file_risks['ell_inf'])
    np.save(os.path.join(dir_name2, f'{prefix}_{attack_mtd}_step{FRM_steps}_run{args.run}.npy'), file_risks)