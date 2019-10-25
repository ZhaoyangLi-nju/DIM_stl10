import torch
from models_STL10_smallsize import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator,get_patch_tensor_from_image_batch
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import statistics as stats
import argparse
import os
from torchvision import transforms
class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        # self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1,7, 7)#26 26

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        # Ej = -F.softplus(-self.global_d(y, M)).mean()
        # Em = F.softplus(self.global_d(y, M_prime)).mean()
        # GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        # return LOCAL + GLOBAL + PRIOR
        return LOCAL +  PRIOR

def adjust_learning_rate(epoch, optimizer,loss_optim):
    LR=2e-4
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    if epoch>60:
        lr=LR*0.1
    if epoch>120:
        lr=LR*0.01
    elif epoch>180:
        lr=LR*0.001
    else:
        lr=LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in loss_optim.param_groups:
        param_group['lr'] = lr
if __name__ == '__main__':


    os.environ["CUDA_VISIBLE_DEVICES"] ='4,5,6,7'
    parser = argparse.ArgumentParser(description='DeepInfomax pytorch')
    parser.add_argument('--batch_size', default=201, type=int, help='batch_size')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    using_crop_patches = True
    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    transform_unlabeled = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
    ])

    train_unlabeled=STL10('/data0/lzy/STL10/data/', split='unlabeled', transform=transform_unlabeled, target_transform=None, download=True)
    # cifar_10_train_dt = CIFAR10(r'c:\data\tv',  download=True, transform=ToTensor())
    print('Train Dataset Length:',len(train_unlabeled))

    train_unlabeled_loader = DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=torch.cuda.is_available())

    encoder = Encoder().to(device)
    loss_fn = DeepInfoMaxLoss().to(device)
    optim = Adam(encoder.parameters(), lr=4e-4)
    loss_optim = Adam(loss_fn.parameters(), lr=4e-4)
    encoder = nn.DataParallel(encoder).to(device)
    loss_fn = nn.DataParallel(loss_fn).to(device)
    torch.backends.cudnn.benchmark = True

    epoch_restart = 0
    root = Path(r'./stl10_7/')
    # root=None
    # if epoch_restart is not None and root is not None:
    #     enc_file = root / Path('encoder' + str(epoch_restart) + '.wgt')
    #     loss_file = root / Path('loss' + str(epoch_restart) + '.wgt')
    #     encoder.load_state_dict(torch.load(str(enc_file)))
    #     loss_fn.load_state_dict(torch.load(str(loss_file)))

    for epoch in range(epoch_restart, 300):

        adjust_learning_rate(epoch,optim,loss_optim)
        batch = tqdm(train_unlabeled_loader, total=len(train_unlabeled) // batch_size)
        train_loss = []
        for x,_ in batch:
            x = x.to(device)
            batch_size=x.shape[0]
            x = get_patch_tensor_from_image_batch(x)#input Batchsize*49,C,H,W,output Batch*49,C,H,W
            # break
            optim.zero_grad()
            loss_optim.zero_grad()
            y, M = encoder(x)

            y  =  y.view(batch_size, -1)
            M = M.view(batch_size,7,7,-1)
            M = M.permute(0, 3, 1, 2)
            # print(y.shape)
            # print(M.shape)
            # rotate images to create pairs for comparison
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            loss = loss_fn(y, M, M_prime)
            loss = loss.mean()
            train_loss.append(loss.item())
            batch.set_description(str(epoch) + ' Loss: ' + str(stats.mean(train_loss[-20:])))
            loss.backward()
            optim.step()
            loss_optim.step()
        if epoch % 5 == 0:
            root = Path('./stl10_7/')
            enc_file = root / Path('encoder' + str(epoch)+'_'+str(round(train_loss[-1],4))+ '.wgt')
            loss_file = root / Path('loss' + str(epoch)+'_'+str(round(train_loss[-1],4))+ '.wgt')
            enc_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), str(enc_file))
            torch.save(loss_fn.state_dict(), str(loss_file))
#