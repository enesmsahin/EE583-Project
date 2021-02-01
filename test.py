from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import random
import joblib

batch_size = 16

mean = joblib.load("./Dataset/Train-dev/mean.joblib")
std = joblib.load("./Dataset/Train-dev/std.joblib")

train_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize(mean, std)])
val_data = torchvision.datasets.ImageFolder("./Dataset/Test2", transform=train_transforms)
valid_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

inv_normalize =  transforms.Normalize(
    mean=-1*np.divide(mean,std),
    std=1/std
)

checkpoint_dir = "./trained_models/lr_0_0005_batch_16_adamw/checkpoint_epoch5.pt"
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)
model.load_state_dict(torch.load(checkpoint_dir))
print("Model and weights are loaded!\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.eval()

total_val_accs = 0
val_accs = []
for batch_idx, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        val_acc = torch.sum(torch.argmax(outputs, 1) == labels) / outputs.shape[0]
        total_val_accs += val_acc

val_accs.append(total_val_accs / (batch_idx + 1))
print('Validation Accuracy: %.3f' %(val_accs[-1]))