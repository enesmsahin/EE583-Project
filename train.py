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

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('./train_logs/')

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

train_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224))])

# Load data to calculate mean and std dev
train_data = torchvision.datasets.ImageFolder("D:/OKUL/6_1/EE583/PROJECT/Dataset/Train-dev", transform=train_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
mean,std = normalization_parameter(train_loader)

# Load data again by also normalizing it
train_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize(mean, std)])
train_data = torchvision.datasets.ImageFolder("D:/OKUL/6_1/EE583/PROJECT/Dataset/Train-dev", transform=train_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = torchvision.datasets.ImageFolder("D:/OKUL/6_1/EE583/PROJECT/Dataset/Validation", transform=train_transforms)
valid_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

inv_normalize =  transforms.Normalize(
    mean=-1*np.divide(mean,std),
    std=1/std
)

classes = train_data.classes
#encoder and decoder to convert classes into integer
decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i
encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]

#plotting rondom images from dataset
# def class_plot(data , encoder ,inv_normalize = None,n_figures = 12):
#     n_row = int(n_figures/4)
#     fig,axes = plt.subplots(figsize=(14, 10), nrows = n_row, ncols=4)
#     for ax in axes.flatten():
#         a = random.randint(0,len(data))
#         (image,label) = data[a]
#         print(type(image))
#         label = int(label)
#         l = encoder[label]
#         if(inv_normalize!=None):
#             image = inv_normalize(image)
        
#         image = image.numpy().transpose(1,2,0)
#         im = ax.imshow(image)
#         ax.set_title(l)
#         ax.axis('off')
#     plt.show()
# class_plot(train_data,encoder, inv_normalize)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

learning_rate = 0.001
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model.train()

num_epochs = 50

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    print("Epoch: ", epoch)

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

        acc = torch.sum(torch.argmax(outputs,1) == labels) / batch_size
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

    total_val_loss = 0.0
    total_val_accs = 0
    for batch_idx, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_losses.append(loss.item())
        val_acc = torch.sum(torch.argmax(outputs, 1) == labels) / batch_size
        total_val_loss += loss.item()
        total_val_accs += val_acc

    val_losses.append(total_val_loss / (batch_idx + 1))
    val_accs.append(total_val_accs / (batch_idx + 1))

    print('[%d] Validation Loss: %.3f \t Validation Accuracy: %.3f' %
                  (epoch, val_losses[-1], val_accs[-1]))
    
    writer.add_scalar('validation_loss', val_losses[-1], epoch)
    writer.add_scalar('validation_accuracy', val_accs[-1], epoch)

    torch.save(model.state_dict(), "./checkpoint_epoch" + str(epoch) + ".pt")
    print("Checkpoint for Epoch " + str(epoch) + " saved")