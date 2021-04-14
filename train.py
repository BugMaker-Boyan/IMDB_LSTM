from model import MyModel
from torch.optim import Adam
from dataset import get_dataloader
import torch.nn.functional as F
import torch
import os
from lib import device
import numpy as np
from tqdm import tqdm


model = MyModel().to(device)
optimizer = Adam(model.parameters(), 0.001)

if os.path.exists("./model/model.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))


def train(epoch):
    data_loader = get_dataloader()
    for idx, (x, target) in enumerate(data_loader):
        optimizer.zero_grad()
        x = x.to(device)
        target = target.to(device)
        output = model(x)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(epoch, idx, loss.item())
        if idx % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")


def test():
    loss_list = []
    acc_list = []
    data_loader = get_dataloader(train=False)
    for idx, (x, target) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Test"):
        x = x.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(x)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu().item())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())
    print("total loss, accuracy:", np.mean(loss_list), np.mean(acc_list))


if __name__ == '__main__':
    for i in range(10):
        train(i)
    test()
