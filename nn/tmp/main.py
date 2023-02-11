import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(3, 2, device="cuda")

    def forward(self, x):
        x = self.linear(x)
        return x


net = Net()
print(net)

params = list(net.parameters())
print("nParams: ", len(params))
print("input size: ", params[0].size())

input = torch.randn(5, 1, 3, device="cuda")
target = torch.randn(5, 1, 2, device="cuda")
criterion = nn.MSELoss()

learning_rate = 0.01
n_iterations = 200
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

for i_iteration in range(n_iterations):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print("loss: ", loss)