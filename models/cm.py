import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum,grad):

        ctx.features = features
        ctx.momentum = momentum
        ctx.grad = grad
        ctx.save_for_backward(inputs, targets,grad)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets,grad = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        if grad:
            #print('grad')
            for x, y in zip(inputs, targets):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()
        
        return grad_inputs, None, None, None,None


def cm(inputs, indexes, features, momentum,grad):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device),torch.tensor(grad).to(inputs.device))

class CM1(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum,grad):

        ctx.features = features
        ctx.momentum = momentum
        ctx.grad = grad
        ctx.save_for_backward(inputs, targets,grad)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets,grad = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        if grad:
            #print('grad')
            for x, y in zip(inputs, targets):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None,None


def cm1(inputs, indexes, features, momentum,grad):
    return CM1.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device),torch.tensor(grad).to(inputs.device))
    
class CM2(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum,grad):

        ctx.features = features
        ctx.momentum = momentum
        ctx.grad = grad
        ctx.save_for_backward(inputs, targets,grad)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets,grad = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        if grad:
            #print('grad')
            for x, y in zip(inputs, targets):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None,None


def cm2(inputs, indexes, features, momentum,grad):
    return CM2.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device),torch.tensor(grad).to(inputs.device))

class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum,grad):

        ctx.features = features
        ctx.momentum = momentum
        ctx.grad = grad
        ctx.save_for_backward(inputs, targets,grad)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets,grad = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        if grad:
            #print('grad')
            batch_centers = collections.defaultdict(list)
            for instance_feature, index in zip(inputs, targets.tolist()):
                batch_centers[index].append(instance_feature)

            for index, features in batch_centers.items():
                distances = []
                for feature in features:
                    distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                    distances.append(distance.cpu().numpy())

                median = np.argmin(np.array(distances))
                ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
                ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None,None


def cm_hard(inputs, indexes, features, momentum,grad):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device),torch.tensor(grad).to(inputs.device))

class CM_Hard1(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum,grad):

        ctx.features = features
        ctx.momentum = momentum
        ctx.grad = grad
        ctx.save_for_backward(inputs, targets,grad)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets,grad = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        if grad:
            #print('grad')
            batch_centers = collections.defaultdict(list)
            for instance_feature, index in zip(inputs, targets.tolist()):
                batch_centers[index].append(instance_feature)

            for index, features in batch_centers.items():
                distances = []
                for feature in features:
                    distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                    distances.append(distance.cpu().numpy())

                median = np.argmin(np.array(distances))
                ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
                ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None,None


def cm_hard1(inputs, indexes, features, momentum,grad):
    return CM_Hard1.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device),torch.tensor(grad).to(inputs.device))

class CM_Hard2(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum,grad):

        ctx.features = features
        ctx.momentum = momentum
        ctx.grad = grad
        ctx.save_for_backward(inputs, targets,grad)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets,grad = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        if grad:
            #print('grad')
            batch_centers = collections.defaultdict(list)
            for instance_feature, index in zip(inputs, targets.tolist()):
                batch_centers[index].append(instance_feature)

            for index, features in batch_centers.items():
                distances = []
                for feature in features:
                    distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                    distances.append(distance.cpu().numpy())

                median = np.argmin(np.array(distances))
                ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
                ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None,None


def cm_hard2(inputs, indexes, features, momentum,grad):
    return CM_Hard2.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device),torch.tensor(grad).to(inputs.device))

class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features1', torch.zeros(num_samples, num_features))
        self.register_buffer('features2', torch.zeros(num_samples, num_features))
        self.register_buffer('features3', torch.zeros(num_samples, num_features))
        self.fids=[]
    def forward(self, inputs, targets,layer,grad=False):

        inputs = F.normalize(inputs, dim=1).cuda()
        targets = targets.cuda()
        if layer == 0:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features1, self.momentum,grad)
            else:
                outputs = cm(inputs, targets, self.features1, self.momentum,grad)
        elif layer == 1:
            if self.use_hard:
                outputs = cm_hard1(inputs, targets, self.features1, self.momentum,grad)
            else:
                outputs = cm1(inputs, targets, self.features2, self.momentum,grad)
        elif layer == 2:
            if self.use_hard:
                outputs = cm_hard2(inputs, targets, self.features1, self.momentum,grad)
            else:
                outputs = cm2(inputs, targets, self.features3, self.momentum,grad)
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss
