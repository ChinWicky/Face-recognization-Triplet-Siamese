# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
#from Network import SiameseNetwork, SiameseDataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import SiameseTriplet
import os
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def split_dataset(directory, split=0.8):
    folders = os.listdir(directory)
    num_train = int(len(folders) * split)
    num_test = int((len(folders) * (1-split))/2)
    random.shuffle(folders)


    train_list = folders[:num_train]
    test_list = folders[num_train:num_train+num_test]
    val_list = folders[num_train+num_test:]

    return train_list, val_list, test_list

# 定义训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        # 将数据和标签放到GPU上
        anchor, positive, negative, label = data
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        output1 = siamese_net(anchor)
        output2 = siamese_net(positive)
        output3 = siamese_net(negative)
        # 计算triplet loss
        loss = criterion(output1, output2, output3)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 打印训练日志
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, running_loss / len(train_loader)))
    return running_loss / len(train_loader)


def test(model, testloader, threshold):
    model.eval()
    correct = 0
    total = 0
    tp = 0
    fn = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            anchor, pos, neg, label = data
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            anchor_embed = model(anchor)
            pos_embed = model(pos)
            neg_embed = model(neg)


            dist_pos = F.pairwise_distance(anchor_embed, pos_embed)
            dist_neg = F.pairwise_distance(anchor_embed, neg_embed)

            # distances = torch.stack([dist_pos, dist_neg], dim=1)
            # labels = torch.zeros(distances.size()[0])
            # labels[dist_pos <= dist_neg] = 1
            # total += labels.size(0)
            # correct += torch.sum(labels == 1).item()
            if  dist_pos < dist_neg + threshold:
                correct += 1
                if i % 2 == 0:
                    tp += 1
                else:
                    tp += 1
            else:
                if i % 2 == 0:
                    fn += 1
                else:
                    fn += 1
            total += 1
    accuracy = correct / total
    recall = tp / (tp + fn)
    print('Accuracy of the network on the test images: {:.2f}%'.format(100 * accuracy))
    print('Recall: {:.2f}%'.format(recall * 100))

    triplet_loss = SiameseTriplet.SiameseTripletLoss(margin=0.2).to(device)
    #loss = triplet_loss(out1, out2, out3)

    # ap_mean = np.mean(pos_scores)
    # an_mean = np.mean(neg_scores)
    # ap_stds = np.std(pos_scores)
    # an_stds = np.std(neg_scores)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (anchor, pos, neg) in enumerate(dataloader):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            anchor_emb, pos_emb, neg_emb = model(anchor, pos, neg)
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    return avg_loss



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')\
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    epochs = 10
    batch_size = 32

    path = './Extracted Faces/Extracted Faces'
    train_list,val_list, test_list = split_dataset(path, 0.8)
    #train_path = './Extracted Faces/Extracted Faces'
    train_dataset = SiameseTriplet.OneShotSiameseDataset(path, train_list, transform=transform, num_samples_per_class=3)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siamese_net = SiameseTriplet.SiameseTripletNetwork().to(device)  # model
    criterion = SiameseTriplet.SiameseTripletLoss(margin=0.2).to(device)

    #criterion = nn.TripletMarginLoss(margin=0.1)
    optimizer = optim.Adam(siamese_net.parameters(), lr=1e-3)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    writer = SummaryWriter(logdir='logs')
    for epoch in range(1, epochs + 1):
        train_loss = train(siamese_net, train_loader, criterion, optimizer, epoch)
        writer.add_scalar('Train Loss', train_loss, epoch)

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': siamese_net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, filename=f'checkpoint_epoch{epoch + 1}.pth.tar')
        #scheduler.step()

    # 关闭SummaryWriter对象
    writer.close()

    test_dataset = SiameseTriplet.OneShotSiameseDataset(path, test_list, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)
    test(siamese_net,test_loader, 0.2)





