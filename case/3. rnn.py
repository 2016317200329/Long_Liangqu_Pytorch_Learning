# Aim: 给定sin波形的前一段，预测后一段曲线
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


num_time_steps = 50
input_size = 1
hidden_size= 16
output_size = 1
lr=0.01
epoch_num = 1000


class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)       # [B,seq, 1],
                                    # 长度为seq的向量上，每个位置只有1个数值

        self.linear = nn.Linear(hidden_size, output_size)

        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
            self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [B,seq,h] --> [seq,h],这里就能看出来RNN的“batch”是同一个timestamp下的数据
        out = out.view(-1,hidden_size)
        out = self.linear(out)              # [seq,h] --> [seq,1]
        out = out.unsqueeze(dim=0)          # [seq,1] --> [seq]

        return out,hidden_prev


########## Init & Training#############
model=Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = torch.zeros([1,1,hidden_size])

for iter in range(epoch_num):
    # start会在sin曲线上随机初始化，以防记住。
    start = np.random.randint(3,size=1)[0]
    # 输入是输入是从start开始的，10个时间戳上的num_time_steps个数据点，
    time_steps = np.linspace(start,start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps,1)       # 符合NN的input

    # x是input，y是target
    # 给的是[0:50)上的输入x，试图预测[1:50]上的结果.【有新的点，有旧的点】
    # 当然也可以给定[0,40),预测[10,50]上的结果
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1,1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1,1)

    output,hidden_prev = model(x,hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output,y)
    model.zero_grad()
    loss.backward()
    # for p in model.parameters():
    #   print(p.grad.norm())
    # torch.nn.utiis.clip_grad_norm(p,10)
    optimizer.step()

    if iter% 100 == 0:
        print(f"Iteration:{iter} f loss={loss.detach().item()}")


# Save
path_name = "rnn.pth"
torch.save(model.state_dict(),"rnn.pth")

######### 1. 我们的方法必须output a vector of "T_i"，接softmax变成[0,1]，然后normalization to [0,1]
######### 2. 但是target data 并不能够包含所有时刻下的, 所以loss还是要分别算。。然后agg.

######## Test ########
Len_to_Pred = 100   ##x.shape[1]

start = np.random.randint(3,size = 1)[0]
# time_steps = np.linspace(start, start + 1010, num_time_steps)
time_steps = np.linspace(start, start + Len_to_Pred, Len_to_Pred)
data = np.sin(time_steps)
print(data.shape)
data = data.reshape(Len_to_Pred,1)


# x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1,1)
x = torch.tensor(data[:-1]).float().view(1, Len_to_Pred - 1,1)
y = torch.tensor(data[1:]).float().view(1, Len_to_Pred - 1,1)

predictions = []
# input其实是相当于给一个start点
input = x[:,0,:]        # x的shape是[1, seq, 1],因此input是[1,1],
print("input shape:", input.shape)
print("x.shape:",x.shape)
# Predict for each timestamp


for _ in range(Len_to_Pred-1):     # For each seq
    input = input.view(1,1,1)
    (pred,hidden_prev) = model(input,hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])

########### Plot ##############
x = x.data.numpy().ravel()
# y = y.data.numpy()
print(predictions)

# Groud Truth

plt.scatter(time_steps[:-1],x.ravel(),s=90)
plt.plot(time_steps[:-1],x.ravel())

# Prediction

plt.scatter(time_steps[1:], predictions)
plt.show()



############# Test的时候？