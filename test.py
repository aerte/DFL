import torch
import torchvision.models as models

vgg16 = models.vgg16(weights='DEFAULT')
print(vgg16)
for param in vgg16.features.parameters():
    param.requires_grad = False
vgg16.classifier[-1] = torch.nn.Linear(4096, 64)
print(vgg16)
