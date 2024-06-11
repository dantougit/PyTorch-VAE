#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-06-11
# @Author  : haozhuolin

import sys

import PIL
import numpy as np
from torchvision import transforms


train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(224),
                                              transforms.ToTensor(),
                                              ])

a = PIL.Image.open('../Data/celeba/img_align_celeba/000001.jpg')

b = train_transforms(a)

print(a)
print(b)

b.save('test.jpg', 'JPEG')

