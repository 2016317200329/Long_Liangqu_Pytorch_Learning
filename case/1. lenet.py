



def main():
    cifar_train datasets.CIFAR10('cifar',True,transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    # wyj: 注意training引用data时的一个trick
    # 为什么要做Normalize: 使(0,1)上的data尽可能的以0为中心
    # 这个是统计出来的cifar上的数据
    transforms.Normalize(mean=[0.185,0.456,0.406],
    srd=[0.229,0.224,0.225])
    ]),download=True)