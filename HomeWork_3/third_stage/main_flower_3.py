# Coding: UTF-8 
# Created by 11 at 2021/1/15
# This "main_flower_3.py" will implement function about: 补充实验

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


# ———————工具函数——————————
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ———————预置参数——————————
loss_record = []
sample_size = 40
epoch_num = 2
batch_size = 4
learn_rate_value = 0.001

# —————————以下为主函数——————————
if __name__ == '__main__':

    # —————————————————迁移学习vgg16模型————————————————————
    net = models.vgg16_bn(pretrained=True)
    # 对迁移模型进行调整
    for parma in net.parameters():
        parma.requires_grad = False
    print(net)
    net.classifier._modules['6'] = nn.Linear(4096, 5)
    # ———————————————————————————————————————————————————————

    transform = transforms.Compose(
        [transforms.Resize([224, 224], interpolation=2), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.ImageFolder(root='./data/mDataset/train/', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root='./data/mDataset/test/', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    classes = ('daisy', 'dandelion', 'rose', 'sunflower', 'tulip')
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learn_rate_value, momentum=0.9)

    for epoch in range(epoch_num):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % sample_size == sample_size - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / sample_size))
                loss_record.append(running_loss / sample_size)
                running_loss = 0.0

    print('Finished Training')
    plt.figure()
    plt.plot(np.array(list(range(len(loss_record)))) * sample_size, loss_record)
    plt.xlabel('迭代次数')
    plt.ylabel('loss函数值')
    plt.title(f'补充实验 loss函数 epoch={epoch_num}')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.savefig(f'补充实验_loss函数_epoch_{epoch_num}.png')
    plt.show()

    data_iter = iter(test_loader)
    images, labels = data_iter.next()

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the all test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if (len(labels) != batch_size):
                break
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(5):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    torch.onnx.export(net, torch.randn(batch_size, 3, 224, 224), 'second_stage_model.onnx', export_params=True,
                      verbose=False,
                      input_names=['input'], output_names=['output'], opset_version=10,
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
