import glob
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import ImageLoader, predict_labels
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
  '''
  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  file_names = glob.glob(os.path.join(dir_name, '*', '*', '*.jpg'))

  scaler = StandardScaler(with_mean=True, with_std=True)
  for file_name in file_names:
    with open(file_name, 'rb') as f:
      img = np.asarray(Image.open(f).convert('L'), dtype='float32')/255.0
      scaler.partial_fit(img.reshape(-1, 1))

  mean = scaler.mean_
  std = scaler.scale_

  return mean, std


def visualize(model: torch.nn.Module,
              split: str,
              data_transforms,
              data_base_path: str = '../data') -> None:
  loader = ImageLoader(data_base_path, split=split, transform=data_transforms)
  class_labels = loader.class_dict
  class_labels = {ele.lower(): class_labels[ele] for ele in class_labels}
  labels = {class_labels[ele]: ele for ele in class_labels}
  paths_and_labels = loader.load_imagepaths_with_labels(class_labels)
  selected = random.choices(paths_and_labels, k=4)
  fig, axs = plt.subplots(2, 2)
  for i in range(4):
    img = loader.load_img_from_path(selected[i][0])
    with torch.no_grad():
      outputs = model(data_transforms(img).unsqueeze(
          0).to(next(model.parameters()).device))
      predicted = predict_labels(outputs).item()
    axs[i//2, i % 2].imshow(img, cmap='gray')
    axs[i // 2, i % 2].set_title('Predicted:{}|Correct:{}'.format(
        labels[predicted], labels[selected[i][1]]))
    axs[i // 2, i % 2].axis('off')
  fig.tight_layout()
  plt.subplots_adjust(wspace=0.5)
  plt.show()
