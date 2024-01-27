# %% [markdown]
# # Parameters

# %%
args= {}
args["metric"]="ms-ssim"
# quality = 1
# args['lmbda'] = 0.0018
# quality = 2
# args['lmbda'] = 0.0035
# quality = 3
# args['lmbda'] = 0.0067
# quality = 4
# args['lmbda'] = 0.0130
quality = 5
args['lmbda'] = 0.0250
# quality = 6
# args['lmbda'] = 0.0483
last_epoch = 0
epochs = 600
args['learning_rate'] = 1e-4
args['aux_learning_rate'] = 1e-3
#args['lmbda'] = 1e-2
args['batch_size'] = 8
args['num_workers']=0
args['clip_max_norm']=1.0
args['root_dir']='our_ssim/q'+str(quality)
args['save']=True
args['patch_size']=256
args['train_root']='Dataset/IQA/natsci/train'
args['val_root']='Dataset/IQA/natsci/val'

# %% [markdown]
# # Import mudoles

# %%
import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from compressai.zoo import cheng2020_anchor
from ipywidgets import interact, widgets
import torch
import torch.nn as nn
from torch.nn import functional as F
from compressai.models.waseda import Cheng2020Anchor
from compressai.models.google import JointAutoregressiveHierarchicalPriors
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3
)
# import warnings
from torchsummary import summary
from compressai.optimizers import net_aux_optimizer
from pytorch_msssim import ms_ssim
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torch.optim as optim
import shutil
import os

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
def save_checkpoint(state, is_best,root_dir, filename="checkpoint.pth.tar"):
    filename = os.path.join(root_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(root_dir,"checkpoint_best_loss.pth.tar"))

# %%
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# %%
class FlexLatLIC(JointAutoregressiveHierarchicalPriors):

    def __init__(self, quality = 1, **kwargs):
        quality_list = {
        1: 128,
        2: 128,
        3: 128,
        4: 192,
        5: 192,
        6: 192,
        }
        N = quality_list[quality]
        super().__init__(N=N, M=N, **kwargs)

        self.g_a_distortion = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_a_clear = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s_distortion = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.maxPool =  torch.nn.MaxPool3d((2,1,1),stride=(2,1,1))
    def forward(self, x):
        y_distortion = self.g_a_distortion(x)
        y_clear = self.g_a_clear(x)
        y_hat_distortion = self.gaussian_conditional.quantize(
            y_distortion, "noise" if self.training else "dequantize"
        )
        y_hat_clear = self.gaussian_conditional.quantize(
            y_clear, "noise" if self.training else "dequantize"
        )

        y_clear_pool = self.maxPool(y_clear)
        y_distortion_pool = self.maxPool(y_distortion)    
        y = torch.cat((y_clear_pool,y_distortion_pool),1)
        
        z = self.h_a(y)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        
        x_hat = self.g_s(y_hat)
        x_hat_distortion = self.g_s_distortion(y_hat_distortion)
        return {
            "x_hat": x_hat,
            "y_hat_clear": y_hat_clear,
            "x_hat_distortion": x_hat_distortion,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    

# %%
class MultiBranchRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, fixed_decoder, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type
        self.fixed_decoder = fixed_decoder

    def forward(self, output, img_target, img_clear, img_distortion):
        N, _, H, W = img_target.size()
        out = {}
        num_pixels = N * H * W

        x_hat = output["x_hat"]
        x_hat_distortion = output["x_hat_distortion"]
        x_hat_clear = self.fixed_decoder(output["y_hat_clear"])

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == ms_ssim:
            l1 = self.metric(x_hat, img_target, data_range=1)
            l2 = self.metric(x_hat_clear, img_clear, data_range=1)
            l3 = self.metric(x_hat_distortion, img_distortion, data_range=1)
            out["mse_loss"] = (3*l1 + l2 + l3)/ 5
            distortion = 1 - out["mse_loss"]
        else:
            l1 = self.metric(x_hat, img_target)
            l2 = self.metric(x_hat_clear, img_clear)
            l3 = self.metric(x_hat_distortion, img_distortion)
            out["mse_loss"] = (3*l1 + l2 + l3)/ 5
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]

# %%
class DACDataset(Dataset):
    def __init__(self, dataset_opt):
        self.rootdir = Path(dataset_opt['root'])
        if not self.rootdir.is_dir():
            raise RuntimeError(f'Invalid directory "{self.rootdir}"')

        # Extract distorted images path
        distorted_path = self.rootdir/'dst'/'*'
        self.samples = []
        for sample in glob.glob(str(distorted_path)):
            self.samples.append(sample)
        self.samples = sorted(self.samples)

        self.phase = dataset_opt['phase']
        self.patch_size = dataset_opt['patch_size']
        
        if self.phase == 'train':
            print("Training")
        elif self.phase == 'val':
            print("Validatin")
        elif self.phase == 'test':
            print("Testing")
        else:
            raise NotImplementedError('wrong phase argument!')            

    def transform(self, ref_image, dst_image):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            ref_image, output_size=(self.patch_size, self.patch_size))
        ref_image = TF.crop(ref_image, i, j, h, w)
        dst_image = TF.crop(dst_image, i, j, h, w)


        # Transform to tensor
        ref_image = TF.to_tensor(ref_image)
        dst_image = TF.to_tensor(dst_image)
        diff_image = torch.abs(ref_image - dst_image)
        # diff_image = TF.to_tensor(diff_image)

        return ref_image, dst_image, diff_image

    def __getitem__(self, index):
        dst_path = self.samples[index]
        ref_name = dst_path.split('/')[-1]
        ref_name = ref_name.split('_')[0]
        if "SCI" in ref_name:
            ref_name = ref_name.upper() +'.bmp'
        else:
            ref_name = ref_name.upper() +'.BMP'
        ref_path = self.rootdir / 'ref' / ref_name
        # print(ref_path)
        noise_image = Image.open(dst_path).convert("RGB")
        ref_image = Image.open(ref_path).convert("RGB")


        ref_image, noise_image, diff_image = self.transform(ref_image, noise_image)

        return ref_image, noise_image, diff_image

    def __len__(self):
        return len(self.samples)


# %%
# The model
net = FlexLatLIC(quality=quality).to(device)
# Load fixed decoder 
cheng_net = cheng2020_anchor(quality=quality, pretrained=True).eval()
g_s_clear = cheng_net.g_s
for param in g_s_clear.parameters():
    param.requires_grad = False   
g_s_clear.to(device)

# %% [markdown]
# # Set traininig parameters

# %%
def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args['learning_rate']},
        "aux": {"type": "Adam", "lr": args['aux_learning_rate']},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

# %%
def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        clear, target, dst = d[0], d[1], d[2]
        target = target.to(device)
        clear = clear.to(device)
        dst = dst.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(target)

        out_criterion = criterion(out_net, target, clear, dst)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            #time.sleep(0.1) 
            print(
                f"Train quality {quality}: ["
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )



# %%
def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            clear, target, dst = d[0], d[1], d[2]
            target = target.to(device)
            clear = clear.to(device)
            dst = dst.to(device)
            out_net = model(target)
            out_criterion = criterion(out_net, target, clear, dst)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

# %%
dataset_opt = {'root': args['train_root'],
               'phase':'train',
               'patch_size':args['patch_size']}
train_set = DACDataset(dataset_opt)
train_dataloader = DataLoader(
        train_set,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

dataset_opt_test = {'root': args['val_root'],
               'phase':'test',
               'patch_size':args['patch_size']}
test_set = DACDataset(dataset_opt_test)
test_dataloader = DataLoader(
        test_set,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

# %%
optimizer, aux_optimizer = configure_optimizers(net, args)
criterion = MultiBranchRateDistortionLoss(g_s_clear,metric=args["metric"], lmbda=args['lmbda'])
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

# %%
best_loss = float("inf")
for epoch in range(last_epoch, epochs):
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    train_one_epoch(
        net,
        criterion,
        train_dataloader,
        optimizer,
        aux_optimizer,
        epoch,
        args['clip_max_norm'],
    )
    loss = test_epoch(epoch, test_dataloader, net, criterion)
    lr_scheduler.step(loss)

    is_best = loss < best_loss
    best_loss = min(loss, best_loss)

    if args['save']:
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best,
            args['root_dir'],
        )


