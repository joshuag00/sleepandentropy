import tonic
import tonic.transforms as transforms
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from tonic import DiskCachedDataset
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn
import random


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 128
num_epochs = 1
num_iters = 50
random.seed(42)


dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
])

trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)


def create_net():
    spike_grad = surrogate.atan()
    beta = 0.5
    net = nn.Sequential(
        nn.Conv2d(2, 12, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Conv2d(12, 32, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 10),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    ).to(device)
    return net


subset_size = int(len(trainset) * 0.1)
subsets = random_split(trainset, [subset_size] * 10)


def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  
    for step in range(data.size(0)):  
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)


def train_net(net, train_subset):
    cached_trainset = DiskCachedDataset(train_subset, cache_path='./cache/nmnist/train')
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(trainloader):
            data, targets = data.to(device), targets.to(device)
            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            if i == num_iters:
                break


def test_net(net, external_testloader):
    mapping_accuracy = {}
    for i in range(10):  
        best_accuracy = 0
        for data, targets in external_testloader:
            data, targets = data.to(device), targets.to(device)
            net.eval()
            spk_rec = forward_pass(net, data)
            acc = SF.accuracy_rate(spk_rec, targets)
            if acc > best_accuracy:
                best_accuracy = acc
        mapping_accuracy[i] = best_accuracy  
    return mapping_accuracy


def group_mappings_by_similarity(mappings, threshold=5):
    unique_mappings = {}
    for i, mapping in enumerate(mappings):
        mapping_frozenset = frozenset(mapping.items())  # Convert to frozenset to make it hashable
        found_group = False
        for key, value in unique_mappings.items():
            # Convert the dictionary key back to a dict for comparison
            if sum(1 for k in mapping if mapping[k] == dict(key).get(k, None)) >= threshold:
                unique_mappings[key].append(i)
                found_group = True
                break
        if not found_group:
            unique_mappings[mapping_frozenset] = [i]  # Use frozenset as key
    
    formatted_output = []
    for idx, (mapping, networks) in enumerate(unique_mappings.items(), start=1):
        formatted_output.append(f"Networks with similar mapping f{idx} = {len(networks)}")
    return formatted_output



nets = [create_net() for _ in range(10)]


class ShuffledLabelNMNIST(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.shuffled_labels = list(range(10))
        random.shuffle(self.shuffled_labels)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample, _ = self.original_dataset[idx]
        shuffled_label = self.shuffled_labels[idx % len(self.shuffled_labels)]
        return sample, shuffled_label

original_testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)
shuffled_testset = ShuffledLabelNMNIST(original_testset)
external_testloader = DataLoader(
    shuffled_testset,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False)
)


mappings = []
for i, (net, subset) in enumerate(zip(nets, subsets)):
    print(f"Training network {i+1}")
    train_net(net, subset)
    print(f"Testing network {i+1} with shuffled labels")
    mapping = test_net(net, external_testloader)
    mappings.append(mapping)
formatted_output = group_mappings_by_similarity(mappings, threshold=5)
for line in formatted_output:
    print(line)
