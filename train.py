import torch
import torch.utils.data as Data
from torchvision import transforms
from torch.optim import lr_scheduler
from models import Two_head
from data_load import Cloth
import os
import copy
import time

data_dir = './dataset'

data_transforms = {'train': transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()]),
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])}

image_dataset = {x: Cloth(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
dataset_size = {x: len(image_dataset[x]) for x in ['train', 'val']}

dataloaders = {x: Data.DataLoader(dataset=image_dataset[x], batch_size=32, shuffle=True, num_workers=4) for x in
               ['train', 'val']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.CrossEntropyLoss()
model = Two_head(512, 3, 3)
model.to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train(epoch):
    running_loss = 0.0
    running_corrects = 0
    for step, (batch_x, batch_color, batch_type) in enumerate(dataloaders['train']):
        batch_x = batch_x.to(device)
        batch_color = batch_color.to(device)
        batch_type = batch_type.to(device)
        with torch.set_grad_enabled(True):
            out_colors, out_types = model(batch_x)
            _, pred_colors = torch.max(out_colors, 1)
            _, pred_types = torch.max(out_types, 1)
            loss = criterion(out_colors, batch_color) + criterion(out_types, batch_type)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        running_corrects += torch.sum((pred_colors == batch_color.data) * (pred_types == batch_type.data))
        if step % 10 == 0:
            print('Epoch: {}\t Step:{}\t Loss:{}'.format(epoch, step, loss.item()))
    epoch_loss = running_loss * 1.0 / dataset_size['train']
    epoch_acc = running_corrects.double() / dataset_size['train']
    print('Train: Epoch: {}\t Loss:{}\t Acc: {}'.format(epoch, epoch_loss, epoch_acc))


def test(epoch):
    running_loss = 0.0
    running_corrects = 0
    model.eval()
    for step, (batch_x, batch_color, batch_type) in enumerate(dataloaders['val']):
        batch_x = batch_x.to(device)
        batch_color = batch_color.to(device)
        batch_type = batch_type.to(device)
        with torch.set_grad_enabled(False):
            out_colors, out_types = model(batch_x)
            _, pred_colors = torch.max(out_colors, 1)
            _, pred_types = torch.max(out_types, 1)
            loss = criterion(out_colors, batch_color) + criterion(out_types, batch_type)

        running_loss += loss.item() * batch_x.size(0)
        running_corrects += torch.sum((pred_colors == batch_color.data) * (pred_types == batch_type.data))

    epoch_loss = running_loss * 1.0 / dataset_size['val']
    epoch_acc = running_corrects.double() / dataset_size['val']
    print('Validation: Epoch: {}\t Loss:{}\t Acc:{}'.format(epoch, epoch_loss, epoch_acc))
    print('-' * 20)
    return epoch_acc


if __name__ == '__main__':
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(20):
        exp_lr_scheduler.step()
        train(epoch)
        acc = test(epoch)
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc:{:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'two_head_pretrain.pkl')
