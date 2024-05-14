import torch
from torch import nn
import torch.nn.functional as F

# wyj: resnet已经属于一个“中型"网络, train 1-2天是可以接收的
class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self,ch_in,ch_out,stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 用大小=1的kernel
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                # wyj: 注意这里stride和外面的stride保持一致，目的是让short cut上和外面达到同样的“长度减半”的效果
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        print(f"before res, data size: {x.shape}")


        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # Short cut: element-wise add
        # x:[b,ch_in,h,w], out:[b,ch_out,h,w]

        res_value = self.extra(x)
        # print(f"after res, the res_value size: {res_value.shape}")
        out = res_value+out
        print(f"after res, out size: {out.shape}")

        return out

class ResNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=3,padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1 = ResBlk(64,128,4)
        self.blk2 = ResBlk(128,256,2)
        self.blk3 = ResBlk(256,512,2)
        self.blk4 = ResBlk(512,512,2)     # wyj: 经验之谈，一般不会上到1024。另外随着channel增大，当feature map降到2*2或者4*4时效果比较好

        self.flatten = nn.Flatten()   # wyj: start_dim=1
        self.outlayer = nn.Linear(512,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        print(f"after blk4: {x.shape}")

        #[b,512,h,w]=>[b,512,1,1]
        # wyj: adaptive表示不管[h,w]多少都可以变成[1,1]的
        x =  F.adaptive_avg_pool2d(x, [1, 1])
        x = torch.flatten(x)
        x = self.outlayer(x)

        return x

def main():
    tmp = torch.randn(2,64,32,32)
    # wyj: 设置stride可以实现H和W的衰减，很有效
    blk = ResBlk(64,128,stride=2)
    out = blk(tmp)
    print('out.shape:',out.shape)

    # x = torch.randn(2, 3, 32, 32)
    # model = ResNet()
    # out= model(x)
    # print('out.shape:',out.shape)
if __name__ == '__main__':
    main()