import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

torch.set_printoptions(sci_mode=False, linewidth=120)

class Housing(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data)
        self.data = self.data.float()
        self.target = torch.from_numpy(target)
        self.target = self.target.float().unsqueeze(-1)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    def __len__(self):
        return len(self.data)

class MLP(nn.Module):
    def __init__(self, in_size=8, out_size=1, hidden_size=512, device="cpu"):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.to(device)

    def forward(self, x):
        x = self.linear(x)
        return x

dataset = fetch_california_housing()
feat_train, feat_test, target_train, target_test = \
    train_test_split(dataset.data, dataset.target, test_size=0.1)
train = np.concatenate((feat_train, target_train[..., np.newaxis]), axis=-1)
np.savetxt('train.csv', train, fmt='%.5f', delimiter=',')
test = np.concatenate((feat_test, target_test[..., np.newaxis]), axis=-1)
np.savetxt('test.csv', test, fmt='%.5f', delimiter=',')
train_dataset = Housing(feat_train, target_train)
train_loader = DataLoader(train_dataset, batch_size=20480, shuffle=True)
test_dataset = Housing(feat_test, target_test)
test_loader = DataLoader(test_dataset, batch_size=20480, shuffle=True)

model = MLP()
optimizer = optim.SGD(model.parameters(), lr=1e-7)
criterion = nn.MSELoss()

print('Starting Training')
for epoch in range(500):
    running_loss = 0.0
    for idx, batch in enumerate(train_loader):
        features, target = batch
        price = model(features)
        loss = criterion(price, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= (idx + 1)
    print(f'epoch: {epoch:03d} batch: {idx:02d} loss: {running_loss:.05f}')

    if epoch % 10 != 0:
        continue

    with torch.no_grad():
        error = 0.0
        for _, batch in enumerate(test_loader):
            features, target = batch
            price = model(features)
            mae = (target - price).abs().sum()
            error += mae.item()
    print(f'epoch: {epoch:03d} batch: {idx:02d} loss: {running_loss:.05f} error: {error:.02f}')
