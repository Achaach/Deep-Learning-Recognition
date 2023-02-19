import copy
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.quantization import DeQuantStub, QuantStub
from torchvision.models import alexnet


class ImageLoader(data.Dataset):
  '''
  Class for data loading
  '''

  train_folder = 'train'
  test_folder = 'test'

  def __init__(self,
               root_dir: str,
               split: str = 'train',
               transform: torchvision.transforms.Compose = None):

    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split

    if split == 'train':
      self.curr_folder = os.path.join(root_dir, self.train_folder)
    elif split == 'test':
      self.curr_folder = os.path.join(root_dir, self.test_folder)

    self.class_dict = self.get_classes()
    self.dataset = self.load_imagepaths_with_labels(self.class_dict)

  def load_imagepaths_with_labels(self,
                                  class_labels: Dict[str, int]
                                  ) -> List[Tuple[str, int]]:

    img_paths = []  # a list of (filename, class index)
    for class_name, class_idx in class_labels.items():
      img_dir = os.path.join(self.curr_folder, class_name, '*.jpg')
      files = glob.glob(img_dir)
      img_paths += [(f, class_idx) for f in files]
    return img_paths

  def get_classes(self) -> Dict[str, int]:
    '''
    Returns:
    -   Dict of class names (string) to integer labels
    '''

    classes = dict()
    classes_list = [d.name for d in os.scandir(self.curr_folder) if d.is_dir()]
    classes = {classes_list[i]: i for i in range(len(classes_list))}
    return classes

  def load_img_from_path(self, path: str) -> Image:
    ''' 
    Args:
    -   path: the path of the image
    Returns:
    -   image: grayscale image loaded using pillow (Use 'L' flag while converting using Pillow's function)
    '''

    img = None
    with open(path, 'rb') as f:
      img = Image.open(f)
      img = img.convert('L')
    return img

  def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
    '''
    Fetches the item (image, label) at a given index

    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    '''
    img = None
    class_idx = None
    filename, class_idx = self.dataset[index]
    # load the image and apply the transforms
    img = self.load_img_from_path(filename)
    if self.transform is not None:
      img = self.transform(img)
    return img, class_idx

  def __len__(self) -> int:
    """
    Returns the number of items in the dataset

    Returns:
        int: length of the dataset
    """
    l = len(self.dataset)
    return l

def get_fundamental_transforms(inp_size: Tuple[int, int],
                               pixel_mean: np.array,
                               pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the core transforms needed to feed the images to our model

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean of the raw dataset [Shape=(1,)]
  - pixel_std: the standard deviation of the raw dataset [Shape=(1,)]
  Returns:
  - fundamental_transforms: transforms.Compose with the fundamental transforms
  '''

  return transforms.Compose([

      transforms.Resize(inp_size),
      transforms.ToTensor(),
      transforms.Normalize(pixel_mean, pixel_std)

  ])

class SimpleNet(nn.Module):
  '''Simple Network with atleast 2 conv2d layers and two linear layers.'''

  def __init__(self):
    '''
    Init function to define the layers and loss function

    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()  # conv2d and supporting layers here
    self.fc_layers = nn.Sequential()  # linear and supporting layers here
    self.loss_criterion = None


    self.cnn_layers = nn.Sequential(
        nn.Conv2d(1, 10, 5),
        nn.MaxPool2d(3, stride = 3),
        nn.ReLU(),
        nn.Conv2d(10, 20, 5),
        nn.MaxPool2d(3, stride = 3),
        nn.ReLU()
        )
    self.fc_layers = nn.Sequential(
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 15)
        )


 
    self.loss_criterion = nn.CrossEntropyLoss(reduction = 'sum')


  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    x = self.cnn_layers(x)
    model_output = x.view(x.size(0), -1)
    model_output = self.fc_layers(model_output)

    return model_output

def predict_labels(model_output: torch.tensor) -> torch.tensor:
  '''
  Predicts the labels from the output of the model.

  Args:
  -   model_output: the model output [Dim: (N, 15)]
  Returns:
  -   predicted_labels: the output labels [Dim: (N,)]
  '''

  predicted_labels = None
  value, predicted_labels = torch.max(model_output, 1)

  return predicted_labels


def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
  '''
  Computes the loss between the model output and the target labels

  Args:
  -   model: model (which inherits from nn.Module), and contains loss_criterion
  -   model_output: the raw scores output by the net [Dim: (N, 15)]
  -   target_labels: the ground truth class labels [Dim: (N, )]
  -   is_normalize: bool flag indicating that loss should be divided by the
                    batch size
  Returns:
  -   the loss value of the input model
  '''
  loss = None

  loss = model.loss_criterion(model_output, target_labels)
  if is_normalize:
    loss /= model_output.size(0)
    

  return loss

def get_optimizer(model: torch.nn.Module,
                  config: dict) -> torch.optim.Optimizer:
  '''
  Returns the optimizer for the model params, initialized according to the config.


  Args:
  - model: the model to optimize for
  - config: a dictionary containing parameters for the config
  Returns:
  - optimizer: the optimizer
  '''

  optimizer = None

  optimizer_type = config["optimizer_type"]
  learning_rate = config["lr"]
  weight_decay = config["weight_decay"]

  if optimizer_type == 'adam':
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
  

  return optimizer

def get_data_augmentation_transforms(inp_size: Tuple[int, int],
                                     pixel_mean: np.array,
                                     pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the data augmentation + core transforms needed to be applied on the train set. Put data augmentation transforms before code transforms. 

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean of the raw dataset
  - pixel_std: the standard deviation of the raw dataset
  Returns:
  - aug_transforms: transforms.compose with all the transforms
  '''

  return transforms.Compose([

      transforms.Resize(inp_size),
      transforms.ColorJitter(),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(pixel_mean, pixel_std)

  ])


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    # Step-1 Load pre-trained alexnet model
    alex = alexnet(pretrained = True)

    # Step-2 Replace the output layer of fully-connected layers.
    self.cnn_layers = alex.features
    self.fc_layers = alex.classifier

    # Step-3 Freezing the layers by setting requires_grad=False
   
    for p in self.cnn_layers.parameters():
      p.requires_grad = False
    
    cnt = 0
    for layer in self.fc_layers:
      cnt += 1
      if cnt < 6:
        for param in layer.parameters():
          param.requires_grad = False

    num_feature = self.fc_layers[6].in_features
    self.fc_layers[6] = nn.Linear(num_feature, 15)


    # Step-4 Assign loss
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
 
    model_output = self.cnn_layers(x)
    model_output = model_output.view(model_output.size(0), -1)
    model_output = self.fc_layers(model_output)

    return model_output



class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None
    
    self.cnn_layers = nn.Sequential(
        nn.Conv2d(1, 10, 5),
        nn.MaxPool2d(3, stride = 3),
        nn.ReLU(),
        nn.Conv2d(10, 20, 5),
        nn.MaxPool2d(3, stride = 3),
        nn.ReLU()
        )
    self.fc_layers = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(500, 100),
        nn.Dropout(p=0.5, inplace=True),
        nn.ReLU(),
        nn.Linear(100, 15)
        )
    
    self.loss_criterion = nn.CrossEntropyLoss(reduction = 'sum')

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None

    x = self.cnn_layers(x)
    model_output = x.view(x.size(0), -1)
    model_output = self.fc_layers(model_output)

    return model_output


class MyAlexNetQuantized(MyAlexNet):
  def __init__(self):
    '''
    Init function to define the layers and loss function.
    '''
    super().__init__()

    self.quant = QuantStub()
    self.dequant = DeQuantStub()

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net.

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None

    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images

    # Step-1: Quantize input using quant()
    #x = self.quant(x)
    #x = self.cnn_layers(x)
    model_output = self.quant(x)
    model_output = self.cnn_layers(model_output)
    
    # Step-2 : Pass the input through the model
    model_output = model_output.reshape((model_output.size(0), -1))
    model_output = self.fc_layers(model_output)
    
    # Step-3: Dequantize output using dequant()
    model_output = self.dequant(model_output)

    return model_output


def quantize_model(float_model: MyAlexNet,
                   train_loader: ImageLoader) -> MyAlexNetQuantized:
  '''
  Quantize the input model to int8 weights.

  Args:
  -   float_model: model with fp32 weights.
  -   train_loader: training dataset.
  Returns:
  -   quantized_model: equivalent model with int8 weights.
  '''

  # copy the weights from original model (still floats)
  quantized_model = MyAlexNetQuantized()
  quantized_model.cnn_layers = copy.deepcopy(float_model.cnn_layers)
  quantized_model.fc_layers = copy.deepcopy(float_model.fc_layers)

  quantized_model = quantized_model.to('cpu')

  quantized_model.eval()

  # Step-1: Set up qconfig of the model
  quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

  # Step-2: Preparing for calibration (use torch.quantization.prepare)
  torch.quantization.prepare_qat(quantized_model, inplace = True)

  # Step-3: Run calibration on the training set
  # (Pass each data in training set to the prepared model)
  for img, _ in train_loader:
    quantized_model(img)

 # Step-4: Do conversion (use torch.quantization.convert)
  torch.quantization.convert(quantized_model, inplace=True)

  quantized_model.eval()

  return quantized_model
