import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from rosettastone.disentanglement import DisentangleNet


def load_cifar10(batch_size=4):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def build_cnn_testbed(is_trained=False, verbose=False, device='cpu'):

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(120, 84)
            self.relu4 = nn.ReLU()
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)
            x = self.relu4(x)
            x = self.fc3(x)
            return x


    net = Net()
    net.to(device)

    # TODO: refactor
    if is_trained:
        
        trainloader, testloader = load_cifar10()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(3):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if verbose and i % 200 == 199:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        if verbose:
            print('Finished Training')

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        
    return net


def test_disentanglement_cnn(device='cpu'):
    net = build_cnn_testbed(is_trained=True, verbose=False, device=device)
    trainloader, _ = load_cifar10()
    dnet = DisentangleNet(net, 'conv2', 15, 'fc1', device=device)
    
    images, _ = next(iter(trainloader))
    images = images.to(device)

    x = dnet.pre_module(images[:1])

    x_alt = dnet.alt_disentanglement_module(x)
    x_alt = dnet.alt_post_module.relu2(x_alt)
    x_alt = dnet.alt_post_module.pool2(x_alt)

    x_alt_unchanged = x_alt.clone()
    x_alt_zeroed = x_alt.clone()
    x_alt_zeroed[:, -1, :, :] = 0

    x_alt_unchanged_flat = dnet.alt_post_module.flatten(x_alt_unchanged)
    x_alt_zeroed_flat = dnet.alt_post_module.flatten(x_alt_zeroed)

    uncopuled_result = dnet.alt_post_module.fc1(x_alt_unchanged_flat)

    decopuled_result = (dnet.alt_post_module.fc1(x_alt_zeroed_flat)
     + dnet.alt_post_module.fc1.weight[:, -25:] @ x_alt_unchanged[:, -1, :, :].flatten())

    assert torch.allclose(uncopuled_result, decopuled_result, atol=1e-6)
