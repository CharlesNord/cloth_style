from data_load import Cloth
from models import Two_head
import torch
import torch.utils.data as Data
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

color_classes = {0: 'black', 1: 'blue', 2: 'red'}
type_classes = {0: 'dress', 1: 'jeans', 2: 'shirt'}


class NewCloth(Cloth):
    def __getitem__(self, item):
        path = self.image_names[item]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        targets = path.split('/')[-2].split('_')
        return img, self.color_classes[targets[0]], self.type_classes[targets[1]], path


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_data = NewCloth(root_dir='./dataset/val', transform=transform)
test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=8)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Two_head(512, 3, 3).to(device)
state_dict = torch.load('./two_head_pretrain.pkl')
model.load_state_dict(state_dict)
model.eval()

wrong_color_img = []
wrong_type_img = []
wrong_color = []
wrong_type = []

wrong_tensor_color = []
wrong_tensor_type = []

for step, (batch_x, batch_color, batch_type, path) in enumerate(test_loader):
    batch_x = batch_x.to(device)
    out_color, out_type = model(batch_x)
    _, pred_color = torch.max(out_color, 1)
    _, pred_type = torch.max(out_type, 1)
    ind_wrong_color = (pred_color.cpu() != batch_color).nonzero()
    ind_wrong_type = (pred_type.cpu() != batch_type).nonzero()
    for idx in ind_wrong_color:
        wrong_color_img.append(path[idx])
        wrong_color.append(color_classes[pred_color[idx].item()])
        wrong_tensor_color.append(batch_x[idx])
    for idx in ind_wrong_type:
        wrong_type_img.append(path[idx])
        wrong_type.append(type_classes[pred_type[idx].item()])
        wrong_tensor_type.append(batch_x[idx])

wrong_img = set(wrong_color_img).union(set(wrong_type_img))
print('Accuracy: {:.4f}'.format(1 - len(wrong_img) * 1.0 / len(test_data)))
print('Wrong color: {}/{}'.format(len(wrong_color_img), len(test_data)))
print('Wrong type: {}/{}'.format(len(wrong_type_img), len(test_data)))

print('Wrong Type:\n')
for i in range(len(wrong_type)):
    print(wrong_type_img[i])
    print(wrong_type[i])


print('Wrong Color:\n')
for i in range(len(wrong_color)):
    print(wrong_color_img[i])
    print(wrong_color[i])


for idx in range(len(wrong_color_img)):
    img = wrong_color_img[idx]
    true_color, true_type = img.split('/')[-2].split('_')
    img = Image.open(img)
    tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)
    # saved_tensor = wrong_tensor_color[idx]
    out_color, out_type = model(tensor)
    _, pred_color = torch.max(out_color, 1)
    _, pred_type = torch.max(out_type, 1)
    pred_color = color_classes[int(pred_color.item())]
    pred_type = type_classes[int(pred_type.item())]
    plt.imshow(img)
    plt.axis('off')
    plt.title('Ground Truth: {}, Pred: {}'.format(true_color + ' ' + true_type, pred_color + ' ' + pred_type))
    plt.show()

for idx in range(len(wrong_type_img)):
    img = wrong_type_img[idx]
    true_color, true_type = img.split('/')[-2].split('_')
    img = Image.open(img)
    tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)
    # saved_tensor = wrong_tensor_type[idx]
    out_color, out_type = model(tensor)
    _, pred_color = torch.max(out_color, 1)
    _, pred_type = torch.max(out_type, 1)
    pred_color = color_classes[int(pred_color.item())]
    pred_type = type_classes[int(pred_type.item())]
    plt.imshow(img)
    plt.axis('off')
    plt.title('Ground Truth: {}, Pred: {}'.format(true_color + ' ' + true_type, pred_color + ' ' + pred_type))
    plt.show()