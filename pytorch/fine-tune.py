from utils import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d
from config import models_genesis_config

config = models_genesis_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(YourDataset, batch_size=config.batch_size, shuffle=True)


# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model, n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(
            self.base_out.size()[0], -1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = self.dense_2(F.relu(self.linear_out))
        return final_out


base_model = unet3d.UNet3D()

# Load pre-trained weights
weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
base_model.load_state_dict(unParalled_state_dict)
target_model = TargetNet(base_model)
target_model.to(device)
target_model = nn.DataParallel(target_model, device_ids=[i for i in range(torch.cuda.device_count())])
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(target_model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model

for epoch in range(intial_epoch, config.nb_epoch):
    scheduler.step(epoch)
    target_model.train()
    for batch_ndx, (x, y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        pred = F.sigmoid(target_model(x))
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
