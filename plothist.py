import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import prepare_partition as pp
import functools

state = np.random.RandomState(69)
data_set = torchvision.datasets.CIFAR10(root='/Users/faerte/Desktop/deep_learning/DFL',train="True", download="False")
target = np.array(data_set.targets)

indices = np.arange(0,len(target),1)
indices2target = np.vstack([indices, target]).T

list_of_indices = pp.build_non_iid_by_dirichlet(random_state=state, indices2targets=indices2target, non_iid_alpha=0.1, num_classes=10, num_indices=len(indices), n_workers=10)

indices = functools.reduce(lambda a, b: a + b, list_of_indices)  # concatenate over the list of indices]
print(np.shape(indices))
labels = target[indices]

sample2 = state.choice(target, size=5000, replace=False)

split_labels = labels[:5000]


plt.figure(figsize=(1, 6))
plt.hist(split_labels, bins=np.arange(11) - 0.5, edgecolor='black', linewidth=1.2, alpha=0.8, color='b', label='non-iid')
plt.hist(sample2, bins=np.arange(11) - 0.5, edgecolor='black', linewidth=1.2, alpha=0.6, color='r', label='iid')
plt.xticks(np.arange(10), data_set.classes, rotation=45)
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Example image distribution of a client')

plt.savefig('test.svg')

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 28})

fig, ax = plt.subplots(figsize=(12,6))
ax.hist(split_labels, bins=np.arange(11) - 0.5, edgecolor='black', linewidth=1.2, alpha=0.8, color='b', label='non-iid')
ax.hist(sample2, bins=np.arange(11) - 0.5, edgecolor='black', linewidth=1.2, alpha=0.6, color='r', label='iid')
ax.set_ylabel('Number of images')
ax.set_xticks(np.arange(10), data_set.classes, rotation=45)
ax.set_title('Example image distribution of a client')
ax.legend()
plt.subplots_adjust(left=0, right=0.9, bottom=0.05, top=0.95)
plt.savefig('plot.svg', bbox_inches='tight')
