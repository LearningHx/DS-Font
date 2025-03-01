"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import time
import collections
from collections import OrderedDict
from .meters import AverageMeter


@torch.no_grad()
def generate_cluster_features(features):
    centers = torch.zeros((len(features),features[0][0].size(1)),device=features[0][0].device)
    labels = collections.defaultdict(list)
    for i,label in enumerate(features):
        same_class_features = features[label]
        centers[i,:] = torch.mean(torch.cat(same_class_features,dim=0),dim=0)
        labels[i] = label
  
    return centers,labels

def extract_features_cluster(model1,model2,dataset,opt):
    model1.eval()   
    model2.eval() 
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features1 = OrderedDict()
    features2 = OrderedDict()
    features3 = OrderedDict()
    nce_layers = [int(i) for i in opt.nce_layers.split(',')]
    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data_time.update(time.time() - end)

            outputs = model2(model1(data['imgs'],nce_layers),nce_layers)
            fids = data['fids']
            for j in range(len(fids)):
                if (int(fids[j])) not in features1:
                    features1[int(fids[j])] = [outputs[0][j].unsqueeze(0)]
                    features2[int(fids[j])] = [outputs[1][j].unsqueeze(0)]
                    features3[int(fids[j])] = [outputs[2][j].unsqueeze(0)]
                else:
                    features1[int(fids[j])].append(outputs[0][j].unsqueeze(0))
                    features2[int(fids[j])].append(outputs[1][j].unsqueeze(0))
                    features3[int(fids[j])].append(outputs[2][j].unsqueeze(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 100== 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(dataset),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
        cluster_centers1,fids = generate_cluster_features(features1)
        cluster_centers2,_ = generate_cluster_features(features2)
        cluster_centers3,_ = generate_cluster_features(features3)
    return [cluster_centers1,cluster_centers2,cluster_centers3],fids

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
