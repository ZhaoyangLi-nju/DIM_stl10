import torch
import torch.nn as nn
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim import Adam
from tensorboardX import SummaryWriter

import statistics as stats
import models_STL10_smallsize as models
from pathlib import Path
from torchvision import transforms
import os
from datetime import  datetime
def precision(confusion):
    correct = confusion * torch.eye(confusion.shape[0])
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    precision = correct / (correct + incorrect)
    total_correct = correct.sum().item()
    total_incorrect = incorrect.sum().item()
    percent_correct = total_correct / (total_correct + total_incorrect)
    return precision, percent_correct
def adjust_learning_rate(epoch, optimizer,LR):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    if epoch>30:
        lr=LR*0.5
    if epoch>60:
        lr=LR*0.1
    elif epoch>90:
        lr=LR*0.01
    elif epoch>120:
        lr=LR*0.001
    else:
        lr=LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] ='4,5,6,7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_classes = 10
    fully_supervised = False
    reload = None
    run_id = 6
    epochs = 150
    LR=1e-2
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    writer = SummaryWriter('/home/lzy/summary/stl10/'+'_'+str(LR)+'_'+str(current_time))
    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    # train_transforms = list()
    # train_transforms.append(SPL10.Resize((96, 96)))
    # train_transforms.append(SPL10.RandomCrop((64, 64)))
    # train_transforms.append(SPL10.RandomHorizontalFlip())
    # train_transforms.append(SPL10.ToTensor())
    # train_transforms.append(SPL10.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                       std=[0.229, 0.224, 0.225]))
    # val_transforms = list()
    #
    # val_transforms.append(SPL10.Resize((96, 96)))
    # val_transforms.append(SPL10.CenterCrop((64,64)))
    # val_transforms.append(SPL10.ToTensor())
    # val_transforms.append(SPL10.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                     std=[0.229, 0.224, 0.225]))
    transform_train = transforms.Compose([
        transforms.Resize(96),
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
    ])
    transform_test= transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
    ])
    train_dataset=STL10('/data0/lzy/STL10/data/', split='train', transform=transform_train, target_transform=None, download=True)
    val_dataset=STL10('/data0/lzy/STL10/data/', split='test', transform=transform_test, target_transform=None, download=True)
    # train_dataset=SPL10.SPL10_Dataset(cfg=None , data_dir='/data0/lzy/STL10/train', transform=transforms.Compose(train_transforms))
    # val_dataset=SPL10.SPL10_Dataset(cfg=None , data_dir='/data0/lzy/STL10/val', transform=transforms.Compose(val_transforms))
    print('Train Dataset Length:',len(train_dataset))
    print('Test Dataset Length:',len(val_dataset))

    # ds = CIFAR10(r'c:\data\tv', download=True, transform=ToTensor()    )
    # len_train = len(ds) // 10 * 9
    # len_test = len(ds) - len_train
    # train_dataset, val_dataset = random_split(ds, [len_train, len_test])
    train_l = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_l = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if fully_supervised:
        classifier = nn.Sequential(
            models.Encoder(),
            models.Classifier()
        ).to(device)
    else:
        classifier = models.DeepInfoAsLatent('run5', '700').to(device)
        # if reload is not None:
        #     classifier = torch.load(r'c:/data/deepinfomax/models/run{run_id}/w_dim{reload}.mdl')

    optim = Adam(classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    reload=0
    for epoch in range(reload + 1, reload + epochs):
        adjust_learning_rate(epoch,optim,LR)
        ll = []
        batch = tqdm(train_l, total=len(train_dataset) // batch_size)
        # for data in batch:
        #     x, target=data['image'],data['label']
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            optim.zero_grad()
            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(str(epoch)+' Train Loss:'+ str(stats.mean(ll)))
            loss.backward()
            optim.step()

        confusion = torch.zeros(num_classes, num_classes)
        batch = tqdm(test_l, total=len(val_dataset)// batch_size)
        ll = []
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(str(epoch) + ' Test Loss:' + str(stats.mean(ll)))

            _, predicted = y.detach().max(1)

            for item in zip(predicted, target):
                confusion[item[0], item[1]] += 1

        precis = precision(confusion)
        writer.add_scalar( 'stl10_precision',precis[1],global_step=epoch)
        print(precis)
        # classifier_save_path = Path('c:/data/deepinfomax/models/run' + str(run_id) + '/w_dim' + str(epoch) + '.mdl')
        # classifier_save_path.parent.mkdir(parents=True, exist_ok=True)
        # torch.save(classifier, str(classifier_save_path))
