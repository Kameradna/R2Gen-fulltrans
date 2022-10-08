# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
#!/usr/bin/env python3
# coding: utf-8
from math import floor
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.models as pymodels
from torchvision import transforms
from torch.utils.data import Dataset

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models
from sklearn import metrics

import bit_common
import bit_hyperrule
import PIL.Image as Image
from torchvision.io import read_image
import os
import json
import csv
import fnmatch


from tqdm import tqdm

class IUXrayDataset(Dataset):#Adapted from NUSdataset and my own work
  #we need init and getitem procedures for a given image
  def __init__(self, data_path, anno_path, train, transforms):
        self.transforms = transforms
        with open(f'{anno_path}/unique_tags_list.json') as f:
          json_data = json.load(f)
        # print(json_data)
        # print(type(json_data))
        self.classes = json_data
        # print(len(self.classes))
        if train:
          anno_path = f'{anno_path}/train.json'
        else:
          anno_path = f'{anno_path}/valid.json'
        with open(anno_path) as fp:
            json_data = json.load(fp)
        
        self.imgs = list(json_data.keys())
        # print(self.imgs)
        self.annos = list(json_data.values())
        # print(type(self.annos))
        each_pos = [0]*len(self.annos[0])
        for sample in range(len(self.annos)):
          each_pos = [each_pos[x]+self.annos[sample][x] for x in range(len(self.annos[sample]))]
        each_neg = [len(self.imgs)-each_pos[x] for x in range(len(each_pos))]
        # print(each_pos)
        # print(len(each_pos))
        # print(each_neg)
        # print(len(each_neg))
        each_pos = [100000000 if each_pos[x] == 0 else each_pos[x] for x in range(len(each_pos))]#really janky workaround for my random sampling of the training set having 0 positive examples of a class, basically just 
        self.pos_weights = [each_neg[x]/each_pos[x] for x in range(len(each_pos))]
        # print(self.pos_weights)
        # print(len(self.pos_weights))
        # print('tick')
        self.data_path = data_path
        for img in range(len(self.imgs)):
          vector = self.annos[img]
          self.annos[img] = np.array(vector, dtype=np.float32) #convert to numpy float vector

  def __getitem__(self, item):
      anno = self.annos[item]
      img_path = os.path.join(self.data_path, self.imgs[item])
      img = Image.open(img_path).convert('RGB')
      if self.transforms is not None:
          img = self.transforms(img)
      return img, anno

  def __len__(self):
      return len(self.imgs)

class CheXpertDataset(Dataset):#Adapted from https://github.com/Stomper10/CheXpert/blob/master/materials.py
  def __init__(self, data_PATH, nnClassCount, policy, split, transform_from_weights, logger):#but what images am I selecting?
    """
    data_PATH: path to the file containing images with corresponding labels.
    Upolicy: name the policy with regard to the uncertain labels.
    """
    image_names = []
    labels = []
    badlabels = 0

    with open(data_PATH, 'r') as f:
      csvReader = csv.reader(f)
      npline = np.array(next(csvReader, None)) #skip first col
      idx = [7, 10, 11, 13, 15, 6, 8, 9, 12, 14, 16, 17, 18, 5] #the key items from the original CheXpertidx = [7, 10, 11, 13, 15]
      label = list(npline[idx])
      self.classes = label
      
      # for idx, item in enumerate(next(csvReader, None)):
      #   print(f"Item {idx} is {item}.")
      for line in tqdm(csvReader): #really hardcore need to work out the policy we will have, it would be sick to report performance on CheXpert in a comparable way to the original
        image_name = line[0] #would be okay to set policy on the ones we know to help, and leave the others, does this code do this?
        npline = np.array(line) #compare to stompers implementation in the rest, especially his lovely AUROC plotting
        # assert len(idx) == len(list(set(idx))) #assert we have not doubled up values
        label = list(npline[idx])
        # raise(NotImplementedError, "What was the output? We want it to include all the different disease markers")
        for i in range(nnClassCount):
          if label[i] != "":
            a = float(label[i])
            # print(a)
            if a == 1:
              label[i] = 1
            elif a == -1:
              if policy == 'diff':
                if i == 1 or i == 3 or i == 4:  # Atelectasis, Edema, Pleural Effusion
                  label[i] = 1                    # U-Ones
                elif i == 0 or i == 2:          # Cardiomegaly, Consolidation
                  label[i] = 0                 # U-Zeroes
                else:
                  label[i] = 1   #for the remaining conditions, uncertain mentions are treated as positive mentions
              elif policy == 'ones':              # All U-Ones
                label[i] = 1
              else:
                label[i] = 0                    # All U-Zeroes
            else: #0 or blank
              label[i] = 0 #if blank
          else:
            label[i] = 0
          # print(label[i])
          # print(type(label[i]))
          # assert isinstance(label[i],int)
        # print(label)
        label = np.array(label,dtype=int)
        if any(label[0:13]) != True:
          badlabels += 1
          label[-1] = 1
                
        image_names.append('data/' + image_name)
        labels.append(label)

    logger.info(f"Cleaned {badlabels} labels")
    self.image_names = image_names
    self.labels = labels
    
    if split == 'train':
        self.transform = transforms.Compose([
          transform_from_weights,
            ])#transforms.RandomHorizontalFlip(), wouldn't this destroy semantic information?
    else:
        self.transform = transform_from_weights

  def __getitem__(self, index):
    '''Take the index of item and returns the image and its labels'''
    image_name = self.image_names[index]
    image = Image.open(image_name).convert('RGB')
    label = self.labels[index]
    image = self.transform(image)
    # print(label)
    return image, torch.FloatTensor(label)

  def __len__(self):
    return len(self.image_names)

    #thank you Stomper10 for your awesome code

def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i

def mktrainval(args, logger):
  """Returns train and validation datasets."""
  #obviously wildly inefficient if you don't have lots of stuff already cached, deal with it?
  possible_weights = torch.hub.load("pytorch/vision", "get_model_weights", name=args.visual_extractor)#relies on build 0.14 of torchvision, may require an updated environment using nightly releases as recommended
  for weights in possible_weights:
      if args.weights == str(weights).split(".")[-1]:#if we are using those weights
          transform_from_weights = weights.transforms()
  if args.dataset == "CheXpert":
    logger.info("Setting up datasets")
    train_set = CheXpertDataset(f"{args.datadir}/train.csv", args.nnClassCount, args.policy, "train", transform_from_weights, logger)
    valid_set = CheXpertDataset(f"{args.datadir}/valid.csv", args.nnClassCount, args.policy, "val", transform_from_weights, logger)
  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")

  micro_batch_size = args.batch // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=True,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader

def run_eval(model, data_loader, device, chrono, logger, args, step, dataset): #consider redoing with Stomper10 in mind
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()
  end = time.perf_counter()

  y_true, y_logits, loss = None, None, None
  for b, (x, y) in enumerate(data_loader):#should be elements of shape (batch size,len(tags))
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)
      # measure data loading time
      chrono._done("eval load", time.perf_counter() - end)
      with chrono.measure("eval fprop"):
        logits = model(x)
        logits.clamp_(0,1)
        c = torch.nn.BCELoss()(logits, y)
        c_num = c.data.cpu().numpy()

        groundtruth = torch.ge(y,0.5)#translates y to tensor
        y_true = groundtruth.cpu().numpy() if isinstance(y_true, type(None)) else np.concatenate((y_true,groundtruth.cpu().numpy()))
        y_logits = logits.cpu().numpy() if isinstance(y_logits, type(None)) else np.concatenate((y_logits,logits.cpu().numpy()))
        loss = c_num if isinstance(loss, type(None)) else np.append(loss,c_num)

    # measure elapsed time
    end = time.perf_counter()
  logger.info(f"Validation loss is {np.mean(loss):.4f}")

  print(f"stats len is {y_true.shape}")

  y_pred = y_logits > 0.5
  y_pred = y_pred.astype(int)
  y_true = y_true.astype(int)

  auroc,precision_, recall_, f1_, support_,accuracy_ = [],[],[],[],[],[]
  for i in range(y_true.shape[1]):
    # print(data_loader.dataset.classes[i])
    # print(y_true[:,i])
    # print(len(y_true[:,i]))
    if any(y_true[:,i]):#if we have positive examples
      auroc.append(metrics.roc_auc_score(y_true[:,i],y_logits[:,i]))
      precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true[:,i],y_pred[:,i],average='binary',zero_division=0)#this batches metrics
      accuracy = metrics.accuracy_score(y_true[:,i],y_pred[:,i])#I think this is exact matches
      precision_.append(precision)
      recall_.append(recall)
      f1_.append(f1)
      support_.append(support)
      accuracy_.append(accuracy)
    # else:
    #   logger.info("No pos values for this class, setting metrics to 1") #causes errors if we do not handle this case, since ROC does not really exist for no positive examples
    #   raise(NotImplementedError)
    #   auroc.append(1.0)
    #   precision_.append(1.0)
    #   recall_.append(1.0)
    #   f1_.append(1.0)
    #   support_.append(None)

  logger.info(f"AUROC = {auroc}")
  logger.info(f"mean AUROC = {np.mean(auroc):.4f}")

  # hamming_mean_loss = metrics.hamming_loss(y_true,y_pred)
  # jaccard_index = metrics.jaccard_score(y_true,y_pred,average='macro')
  # average_precision = metrics.average_precision_score(y_true,y_pred,average='macro')

  #RocCurveDisplay.from_predictions(y_true,y_pred)
  # metrics.PrecisionRecallDisplay(precision,recall,pos_label=[what have you]
  # label_cardinality = np.sum(support)/len(dataset)
  # label_density = np.sum(support)/len(dataset)/len(dataset.classes)

  logger.info(f"Validation@{step}, "
              f"Mean_loss={np.mean(loss):.4f}, "
              f"Mean_precision={np.mean(precision_):.2%}, "
              f"Mean_recall={np.mean(recall_):.2%}, "
              f"Mean_accuracy={np.mean(accuracy_):.2%}, "
              f"Mean_F1 score={np.mean(f1_):.2%}, "

              f"AUROC={np.mean(auroc):.5f}, "
              # f"AUPRC={average_precision:.5f}, "
              )
  logger.flush()
  model.train()
  return np.mean(auroc), np.mean(f1_)

def main(args):
  logger = bit_common.setup_logger(args)
  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  
  torch.backends.cudnn.benchmark = True
  # scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)
  
  model = getattr(pymodels, args.visual_extractor)(weights=args.weights)

  if fnmatch.fnmatch(args.visual_extractor,"*resnet*"):
      num_features = model.fc.in_features
      model.fc = nn.Linear(num_features, len(valid_set.classes),bias=True)
  elif fnmatch.fnmatch(args.visual_extractor,"vit*"):
      num_features = model.heads.head.in_features
      model.heads.head = nn.Linear(num_features, len(valid_set.classes),bias=True)
  elif fnmatch.fnmatch(args.visual_extractor,"swin*"):
      num_features = model.head.in_features
      model.head = nn.Linear(num_features, len(valid_set.classes),bias=True)
      # print(model)
  # elif fnmatch.fnmatch(args.visual_extractor,"alexnet"):
  #     modules = list(model.children())[:-2]
  #     args.model = nn.Sequential(*modules)
  # elif fnmatch.fnmatch(args.visual_extractor,"regnet*"):
  #     modules = list(model.children())[:-2]
  #     args.model = nn.Sequential(*modules)
  # elif fnmatch.fnmatch(args.visual_extractor,"densenet*"): #inspiration from stomper time
  #     args.model = model
  #     raise(NotImplementedError)
  else:
      print(model)
      print(f"we have not implemented the {args.visual_extractor} visual extractor for this paper")
      raise(NotImplementedError)


  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)

  step = 0
  best_mean_auc = 0
  best_mean_f1 = 0

  # Note: no weight-decay!
  if args.optim == "Adam":
    logger.info("Using Adam")
    optim = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999)) #*maybe lr is wrong*"
  elif args.optim == "SGD":  
    logger.info("Using SGD")
    optim = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
  else:
    raise(NotImplementedError, "Optimiser you chose was not found")

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "bit.pth.tar")
  try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at step {step}")
  except FileNotFoundError:
    logger.info("Fine-tuning from BiT")

  model = model.to(device)
  optim.zero_grad()

  model.train()
  cri = torch.nn.BCELoss().to(device) #pos_weight=torch.Tensor(train_set.pos_weights)

  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  # mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.perf_counter()

  step_name = '0'
  run_eval(model, valid_loader, device, chrono, logger, args, step_name, valid_set)

  with lb.Uninterrupt() as u:
    for x, y in recycle(train_loader):
      # measure data loading time, which is spent in the `for` statement.
      chrono._done("load", time.perf_counter() - end)

      if u.interrupted:
        break


      # with torch.cuda.amp.autocast(enabled=args.use_amp): #MY ADDITION
      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # Update learning-rate, including stop training if over.
      lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr

      with chrono.measure("fprop"):
        logits = model(x)
        logits.clamp_(0,1)
        c = cri(logits, y)
        c_num = float(c.data.cpu().numpy()) # Also ensures a sync point.

      # Accumulate grads
      with chrono.measure("grads"):
        # scaler.scale(c / args.batch_split).backward()#MY ADDITION
        (c/args.batch_split).backward()#torch.ones_like(c) if reduction='none'
        accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
      logger.flush()

      # Update params
      if accum_steps == args.batch_split:
        with chrono.measure("update"):
          optim.step()
          # scaler.step(optim)#MY ADDITION
          # scaler.update()#MY ADDITION
          optim.zero_grad(set_to_none=True)#my edit
        step += 1
        accum_steps = 0

        #eval every?

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          #save best AUC
          mean_auc, mean_f1 = run_eval(model, valid_loader, device, chrono, logger, args, step, valid_set)
          if mean_auc > best_mean_auc or mean_f1 > best_mean_f1:
            print("BIG MONEY BIG MONEY BIG MONEY BIG MONEY")
            best_mean_auc = mean_auc
            best_mean_f1 = mean_f1
            #delete last best save or use deepcopy()
            savename = pjoin(args.logdir, args.name, f"{best_mean_auc}_{best_mean_f1}_{step}bit.pth.tar")
            best_model_wts = copy.deepcopy(model.state_dict())
            if args.save:
              quicksave_model = copy.deepcopy(model.state_dict())
              model.load_state_dict(best_model_wts)
              torch.save({
                  "step": step,
                  "model": model.state_dict(),
                  "optim" : optim.state_dict(),
              }, savename)
              model.load_state_dict(quicksave_model)

      end = time.perf_counter()

    # Final eval at end of training.
    step_name = 'end'
    run_eval(model, valid_loader, device, chrono, logger, args, step_name, valid_set)

  logger.info(f"Timings:\n{chrono}")


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  # parser.add_argument("--use_amp", dest="use_amp",action="store_true",
  #                    help="Use Automated Mixed Precision to save potential memory and compute?")
  parser.add_argument("--annodir", required=True, help="Where are the annotation files to load?")
  # parser.add_argument("--chexpert", dest="chexpert", action="store_true",help="Run as the chexpert paper?")
  parser.add_argument("--pretrained", dest="pretrained", action="store_true",help="Do you want a pretrained network?")
  main(parser.parse_args())
