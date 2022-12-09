import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from myNets.squeezenet import squeezenet1_0  # 可更换为其他

"""
__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']
"""


train_root = './datas/train/'
test_root = './datas/test/'
# 将文件夹的内容载入dataset
train_dataset = torchvision.datasets.ImageFolder(root=train_root, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root=test_root, transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)


model = squeezenet1_0(num_classes=5)
learning_rate = 0.1
batch_size = 64
epochs = 100
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # pred_probab = nn.Softmax(dim=1)(pred)
            # y_pred = pred_probab.argmax(1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
