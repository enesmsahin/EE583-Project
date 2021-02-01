from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
import joblib

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('./training_results/')

def normalization_parameter(dataloader):
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    for data,_ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(),std.numpy()

batch_size = 16

getTrainMeanStd = True

if getTrainMeanStd:
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((224,224))])

    # Load data to calculate mean and std dev
    train_data = torchvision.datasets.ImageFolder("./Dataset/Train-dev", transform=train_transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    mean,std = normalization_parameter(train_loader)

    print("Saving mean and std of training data to the disk!\n")
    joblib.dump(mean, "./Dataset/Train-dev/mean.joblib")
    joblib.dump(std, "./Dataset/Train-dev/std.joblib")
else:
    print("Loading mean and std of training data from the disk!\n")
    mean = joblib.load("./Dataset/Train-dev/mean.joblib")
    std = joblib.load("./Dataset/Train-dev/std.joblib")

# Load data again by also normalizing it
train_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize(mean, std)])
train_data = torchvision.datasets.ImageFolder("./Dataset/Train-dev", transform=train_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = torchvision.datasets.ImageFolder("./Dataset/Validation", transform=train_transforms)
valid_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

inv_normalize =  transforms.Normalize(
    mean=-1*np.divide(mean,std),
    std=1/std
)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

learning_rate = 0.0005
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 50

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    print("Epoch: ", epoch)

    model.train()

    running_loss = 0.0
    running_corrects = 0
    iterator = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        iterator += 1
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = torch.sum(torch.argmax(outputs,1) == labels) / outputs.shape[0]
        running_corrects = running_corrects + acc
        train_losses.append(loss.item())
        train_accs.append(acc)
        
        # print statistics
        running_loss += loss.item()
        if batch_idx % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] Train Loss: %.3f \t Train Accuracy: %.3f' %
                  (epoch, batch_idx + 1, running_loss / iterator, running_corrects / iterator))
            
            writer.add_scalar('training_loss', running_loss / iterator, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('training_accuracy', running_corrects / iterator, epoch * len(train_loader) + batch_idx)

            running_loss = 0.0
            running_corrects = 0
            iterator = 0

    model.eval()

    total_val_loss = 0.0
    total_val_accs = 0
    for batch_idx, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_losses.append(loss.item())
        val_acc = torch.sum(torch.argmax(outputs, 1) == labels) / outputs.shape[0]
        total_val_loss += loss.item()
        total_val_accs += val_acc

    val_losses.append(total_val_loss / (batch_idx + 1))
    val_accs.append(total_val_accs / (batch_idx + 1))

    print('[%d] Validation Loss: %.3f \t Validation Accuracy: %.3f' %
                  (epoch, val_losses[-1], val_accs[-1]))
    
    writer.add_scalar('validation_loss', val_losses[-1], epoch)
    writer.add_scalar('validation_accuracy', val_accs[-1], epoch)

    torch.save(model.state_dict(), "./training_results/checkpoint_epoch" + str(epoch) + ".pt")
    print("Checkpoint for Epoch " + str(epoch) + " saved")