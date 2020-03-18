# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


# Weighted variant
def weighted_smooth_l1_loss(input, target, weight, beta=1. / 9):
    numel = input.numel()
    #print("input:", input.shape)
    #print("numel:", numel)
    #print("target:", target.shape)
    #print("weight:", weight.shape)
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    #print("loss:", loss.shape)
    loss = torch.sum(loss, dim=1)
    #print("loss:", loss, loss.shape)
    weighted_loss = loss * weight
    #print("weighted_loss:", weighted_loss, weighted_loss.shape)
    #return weighted_loss.sum() / numel
    return weighted_loss.sum()

