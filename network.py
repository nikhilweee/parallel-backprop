import torch
import numpy as np
from torch import nn, optim

torch.set_printoptions(sci_mode=False, linewidth=120)
torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self, in_size=8, out_size=1, hidden_size=512, device="cpu"):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.to(device)

    def forward(self, x):
        x = self.linear(x)
        return x

train = np.loadtxt('train.csv', delimiter=',')
test = np.loadtxt('test.csv', delimiter=',')
feat_train, target_train = np.split(train, [8], axis=1)
feat_test, target_test = np.split(test, [8], axis=1)

feat_train = torch.from_numpy(feat_train).float()
target_train = torch.from_numpy(target_train).float()
feat_test = torch.from_numpy(feat_test).float()
target_test = torch.from_numpy(target_test).float()

print(tuple(feat_train.shape))

model = MLP()
optimizer = optim.SGD(model.parameters(), lr=1e-7)
criterion = nn.MSELoss()

for epoch in range(101):
    price = model(feat_train)
    loss = criterion(price, target_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss = loss.item()

    if epoch > 10 and epoch % 10 != 0:
        continue

    print(f'epoch: {epoch:04d} loss: {running_loss:.05f}')


    # with torch.no_grad():
    #     error = 0.0
    #     price = model(feat_test)
    #     mae = (target_test - price).abs().sum()
    #     error += mae.item()
    # print(f'epoch: {epoch:04d} loss: {running_loss:.05f} error: {error:.02f}')
