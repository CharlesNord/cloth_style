import torch
from torchvision import models
from tensorboardX import SummaryWriter


class Two_head(torch.nn.Module):
    def __init__(self, in_feature, out_feature1, out_feature2):
        super(Two_head, self).__init__()
        self.head1 = torch.nn.Linear(in_feature, out_feature1)
        self.head2 = torch.nn.Linear(in_feature, out_feature2)
        resnet = models.resnet18(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[0:-1])

    def forward(self, x):
        feature = self.resnet(x)
        feature = feature.view(feature.size(0), -1)
        out1 = self.head1(feature)
        out2 = self.head2(feature)
        return out1, out2


if __name__ == '__main__':
    dummy_input = torch.rand(1, 3, 224, 224)
    with SummaryWriter(log_dir='./log', comment='two-head') as w:
        model = Two_head(512, 3, 3)
        w.add_graph(model, dummy_input)
