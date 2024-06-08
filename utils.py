import os
import pickle
from PIL import Image
import numpy as np
import torch
import shutil
from torch.nn.modules.container import T
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object): # summary average of object

  def __init__(self):
    self.reset() # reset()

  def reset(self): # init(reset)
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1): # get average
    self.sum += val * n 
    self.cnt += n
    self.avg = self.sum / self.cnt

# summary accuracy betwwen outout and target
# topk <= summary predict of high k 
def accuracy(output, target, topk=(1,)): # (1,) <= means top 1
  maxk = max(topk)
  batch_size = target.size(0)
    # output <= before init SoftMax
    # target <= real label
  _, pred = output.topk(maxk, 1, True, True) # _ <= none <= method, pred <= index
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct <= Bool tensor
  res = []
  for k in topk: # loop
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

# cutout some area of image to strong training data
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2) # get heigh, width
        mask = np.ones((h, w), np.float32) # make mask 
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
    # para = 0.0
    # for name, v in model.named_parameters():
    #     if v.requires_grad == True:
    #         if "auxiliary" not in name:
    #             para += np.prod(v.size())
    # return para / 1e6
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6



def save_checkpoint(state, is_best, save):
  # state <= dict of state of model
  # is_best => if best <= True else <= False
  # save => path of dir to save model
  filename = os.path.join(save, 'checkpoint.pth.tar')

  torch.save(state, filename)
  # if is_best is True save another name
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

# to do Dropout
def drop_path(x, drop_prob):
  # x : input data
  # drop_prob : probobility of dropout

  if drop_prob > 0.:
    # summary of being remain probability
    keep_prob = 1.-drop_prob
    # make mask
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    # division input data 
    x.div_(keep_prob)
    # multiple mask to apply Dropout 
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
    
  if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'),exist_ok=True)
    
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
      
    
# save evaluate image
def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')
   
def save_images_ycbcr(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = ycbcr2rgb(image_numpy)
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')
    
# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def ycbcr2rgb(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2rgb function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): RGB image array data

    """
    #image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    #image = image.astype(image_dtype)

    return image
  
# save best model
def save_best(model, is_best, path):
    filename = os.path.join(path, 'backup_model.pt')
    torch.save(model.state_dict(), filename)
    if is_best:
      best_filename = os.path.join(path, 'best_model.pt')
      shutil.copyfile(filename, best_filename)
      print("find best_model")


# find latest epoch
def find_latest_epoch(model_path):
    model_files = [f for f in os.listdir(model_path) if f.startswith('weights_') and f.endswith('.pt')]
    if not model_files:
        return None  # No model files found

    epoch_numbers = [int(f.split('_')[1].split('.')[0]) for f in model_files]
    latest_epoch = max(epoch_numbers)
    return latest_epoch