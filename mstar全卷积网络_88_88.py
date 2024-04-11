import torch
import torchvision
from torch import nn
from torch.utils.data import TensorDataset  # TensorDataset可以用来对tensor数据进行打包,该类中的 tensor 第一维度必须相等(即每一个图片对应一个标签)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 从mat格式文件中读取numpy数据并转换为tensor格式（方法！！！）
data88 = scipy.io.loadmat('./mstar_data/88_88.mat')
train_data = data88['train_data']
train_label = data88['train_labels']
test_data = data88['test_data']
test_label = data88['test_labels']
print(train_data.shape, train_label.shape, test_data.shape,
      test_label.shape)  # (3671, 88, 88) (1, 3671) (3203, 88, 88) (1, 3203)
train_data = train_data.reshape(3671, 1, 88, 88)  # 灰度图像
test_data = test_data.reshape(3203, 1, 88, 88)
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
train_data, train_label, test_data, test_label = map(torch.tensor, (
train_data, train_label.squeeze(), test_data, test_label.squeeze()))    # numpy转tensor
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

# 数据集标准化
mean = train_data.mean()
std = train_data.std()
print(mean, std)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
print(train_data.mean(), train_data.std(), test_data.mean(), test_data.std())


# 绘图函数和绘制曲线函数
def plot_curve(data, name, leg):
    fig = plt.figure()
    plt.plot(range(len(data)), data, 'blue')
    plt.legend([leg], fontsize=14, loc='best')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(name, fontsize=14)
    plt.grid()
    plt.savefig('fig' + leg)
    plt.show()


def plot_image(img, label, img_name, clas):
    fig = plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # plt.tight_layout()
        plt.imshow(img[i].view(88, 88), cmap='gray')
        plt.title("{}:{}".format(img_name, label[i].item()))  # item()的作用是取出tensor格式的元素值并返回该值，保持原元素类型不变
        plt.xticks([])
        plt.yticks([])
    plt.savefig('fig' + clas)
    plt.show()


# 训练集10种样本可视化
train_data09 = []
train_label09 = []
for i in range(10):
    train_data09.append(train_data[train_label == i][0])  # 这里要再加[0]才能正常显示label=i时的第一组图片
    train_label09.append(torch.tensor(i))  # 因为此时train_label就等于i，但train_label是tensor格式，因此需要将i转换为tensor格式
print(len(train_data09), train_label09)
plot_image(train_data09, train_label09, 'label', '88_88')

# print('aaa')
# print(train_label.shape)

# 加载数据集
train_xy = TensorDataset(train_data, train_label)  # 相当于将图片和对应的标签打包在了一起
test_xy = TensorDataset(test_data, test_label)
train = DataLoader(train_xy, batch_size=64, shuffle=True)  # shuffle=True用于打乱数据集，每次都会以不同的顺序返回
test = DataLoader(test_xy, batch_size=256)
print(len(train_xy), len(train), len(test_xy), len(test))


class mstar88_cnn(nn.Module):  # n=88
    def __init__(self):
        super(mstar88_cnn, self).__init__()
        self.model_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0),  # n=84
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 42

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),  # 38
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 19

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=1, padding=0),  # 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7

            nn.Dropout(0.5),  # 这里用Dropout2d应该也可以
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),  # 3
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=3, stride=1, padding=0),  # 1
            nn.BatchNorm2d(10)
        )

    def forward(self, x):
        x = self.model_cnn(x)
        x = x.view(x.shape[0], -1)  # 矩阵的每一行就是这个批量中每张图片的各个参数，即矩阵中一行对应一张图片
        x = F.softmax(x, 1)
        return x


model = mstar88_cnn()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
learning_rate = 0.001
# optimizer=torch.optim.SGD(params=model.parameters(),lr=learning_rate,momentum=0.9)

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []
epochs = 10

for epoch in range(epochs):

    if (epoch + 1) == 50:
        learning_rate = learning_rate * 0.9
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)

    print("-----第{}轮训练开始------".format(epoch + 1))
    train_loss = 0.0
    test_loss = 0.0
    train_sum, train_cor, test_sum, test_cor = 0, 0, 0, 0

    # 训练步骤开始
    model.train()
    for batch_idx, data in enumerate(train):
        inputs, labels = data
        inputs, labels=inputs.to(device),labels.to(device)
        labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
        optimizer.zero_grad()
        outputs = model(inputs.float())  # 需要加.float()，否则会报错
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算每轮训练集的Loss
        train_loss += loss.item()

        # 计算每轮训练集的准确度
        _, predicted = torch.max(outputs.data, 1)  # 选择最大的（概率）值所在的列数就是他所对应的类别数，
        train_cor += (predicted == labels).sum().item()  # 正确分类个数
        train_sum += labels.size(0)  # train_sum+=predicted.shape[0]

    # 测试步骤开始
    model.eval()
    # with torch.no_grad():
    for batch_idx1, data in enumerate(test):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        labels = torch.tensor(labels, dtype=torch.long)
        outputs = model(inputs.float())
        loss = loss_fn(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_cor += (predicted == labels).sum().item()
        test_sum += labels.size(0)

    print("Train loss:{}   Train accuracy:{}%   Test loss:{}   Test accuracy:{}%".format(train_loss / batch_idx,
                                                                                         100 * train_cor / train_sum,
                                                                                         test_loss / batch_idx1,
                                                                                         100 * test_cor / test_sum))
    train_loss_list.append(train_loss / batch_idx)
    train_acc_list.append(100 * train_cor / train_sum)
    test_acc_list.append(100 * test_cor / test_sum)
    test_loss_list.append(test_loss / batch_idx1)

# 保存网络
torch.save(model, "mstar_88_epoch{}.pth".format(epochs))

fig = plt.figure()
plt.plot(range(len(train_loss_list)), train_loss_list, 'blue')
plt.plot(range(len(test_loss_list)), test_loss_list, 'red')
plt.legend(['Train Loss', 'Test Loss'], fontsize=14, loc='best')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid()
plt.savefig('figLOSS_88')
plt.show()

fig = plt.figure()
plt.plot(range(len(train_acc_list)), train_acc_list, 'blue')
plt.plot(range(len(test_acc_list)), test_acc_list, 'red')
plt.legend(['Train Accuracy', 'Test Accuracy'], fontsize=14, loc='best')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy(%)', fontsize=14)
plt.grid()
plt.savefig('figAccuracy_88')
plt.show()
