from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn


def get_patch_tensor_from_image_batch(img_batch):

    # Input of the function is a tensor [B, C, H, W]
    # Output of the functions is a tensor [B * 49, C, 64, 64]

    patch_batch = None
    all_patches_list = []

    for y_patch in range(7):
        for x_patch in range(7):

            y1 = y_patch * 8
            y2 = y1 + 16

            x1 = x_patch * 8
            x2 = x1 + 16

            img_patches = img_batch[:,:,y1:y2,x1:x2] # Batch(img_idx in batch), channels xrange, yrange
            img_patches = img_patches.unsqueeze(dim=1)
            all_patches_list.append(img_patches)

            # print(patch_batch.shape)
    all_patches_tensor = torch.cat(all_patches_list, dim=1)

    patches_per_image = []
    for b in range(all_patches_tensor.shape[0]):
        patches_per_image.append(all_patches_tensor[b])

    patch_batch = torch.cat(patches_per_image, dim = 0)
    return patch_batch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 96, kernel_size=3, stride=1)
        self.c1 = nn.Conv2d(96, 192, kernel_size=3, stride=1)
        self.c2 = nn.Conv2d(192, 384, kernel_size=3, stride=1)
        self.c3 = nn.Conv2d(384, 384, kernel_size=3, stride=1)
        self.c4 = nn.Conv2d(384, 192, kernel_size=3, stride=1)

        self.l1 = nn.Linear(6912, 4096)#20*20
        self.l2 = nn.Linear(4096, 64)#20*20

        self.b0 = nn.BatchNorm2d(96)
        self.b1 = nn.BatchNorm2d(192)
        self.b2= nn.BatchNorm2d(384)
        self.b3= nn.BatchNorm2d(384)
        self.b4= nn.BatchNorm2d(192)

        self.bl1=nn.BatchNorm1d(4096)
        self.bl2=nn.BatchNorm1d(64)

        # self.MaxPool=nn.MaxPool2d(3,2)



    def forward(self, x):
        h = F.relu(self.b0(self.c0(x)))
        features = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(features)))
        h = F.relu(self.b3(self.c3(h)))
        h = F.relu(self.b4(self.c4(h)))
        encoded_0 = F.relu(self.bl1(self.l1(h.view(x.shape[0], -1))))
        encoded = F.relu(self.bl2(self.l2(encoded_0)))

        return encoded, features


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(192, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(3724, 512)#32 * 22 * 22 + 64
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(30784, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(3136, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3136, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.l2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        clazz = F.relu(self.bn1(self.l1(encoded)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz


class DeepInfoAsLatent(nn.Module):
    def __init__(self, run, epoch):
        super().__init__()
        # model_path = Path(r'/home/lzy/DeepInfomaxPytorch/') / Path(str(run)) / Path('encoder' + str(epoch) + '.wgt')
        self.encoder = Encoder()
        self.encoder = nn.DataParallel(self.encoder).cuda()
        # for k,v in self.encoder.state_dict().items():
        #   print(k)
        #   print(v)
        #   break
        self.encoder.load_state_dict(torch.load(str('/home/lzy/DeepInfomaxPytorch/stl10_7/encoder80_0.1368.wgt')))

        self.classifier = Classifier()

    def forward(self, x):
        batch_size=x.shape[0]
        x = get_patch_tensor_from_image_batch(x)
        z, features = self.encoder(x)
        z = z.view(batch_size, -1)
        # print('z:',z.shape)
        z = z.detach()
        return self.classifier((z, features))