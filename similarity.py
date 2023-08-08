import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def cosine_sim(x, y):
    """Cosine similarity between all the image and sentence pairs. Assumes that x and y are l2 normalized"""
    return x.mm(y.t())

class MPsimilarity(nn.Module):
    def __init__(self, avg_pool):
        super(MPsimilarity, self).__init__()
        self.avg_pool = avg_pool
        self.alpha, self.beta = nn.Parameter(torch.ones(1)).cuda(), nn.Parameter(torch.zeros(1)).cuda()
        
    def forward(self, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs)
        avg_similarity = self.avg_pool(torch.sigmoid(self.alpha * dist.unsqueeze(0) + self.beta)).squeeze(0)
        return avg_similarity
    
class SetwiseSimilarity(nn.Module):
    def __init__(self, img_set_size, txt_set_size, denominator, temperature=1):
        super(SetwiseSimilarity, self).__init__()
        # poolings
        self.img_set_size = img_set_size
        self.txt_set_size = txt_set_size
        self.denominator = denominator
        self.temperature = temperature
        
        self.xy_max_pool = torch.nn.MaxPool2d((self.img_set_size, self.txt_set_size))
        self.xy_avg_pool = torch.nn.AvgPool2d((self.img_set_size, self.txt_set_size))
        self.x_axis_max_pool = torch.nn.MaxPool2d((1, self.txt_set_size))
        self.x_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.txt_set_size))
        self.y_axis_max_pool = torch.nn.MaxPool2d((self.img_set_size, 1))
        self.y_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.img_set_size, 1))
        
        self.mp_dist = MPsimilarity(self.xy_avg_pool)
        
    def smooth_chamfer_similarity_euclidean(self, img_embs, txt_embs):
        """
            Method to compute Smooth Chafer Similarity(SCD). Max pool is changed to LSE.
            Use euclidean distance(L2-distance) to measure similarity between elements.
        """
        dist = torch.cdist(img_embs, txt_embs)
        
        first_term = -self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(-self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        second_term = -self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(-self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (first_term / (self.img_set_size * self.temperature) + second_term / (self.txt_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
    def smooth_chamfer_similarity_cosine(self, img_embs, txt_embs):
        """
            cosine version of smooth_chamfer_similarity_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        first_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        second_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (
            first_term / (self.img_set_size * self.temperature) +\
            second_term / (self.txt_set_size * self.temperature)
        ) / (self.denominator)

        return smooth_chamfer_dist
    
    def chamfer_similarity_cosine(self, img_embs, txt_embs):
        """
            cosine version of chamfer_similarity_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        first_term = self.y_axis_sum_pool(self.x_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        second_term = self.x_axis_sum_pool(self.y_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (first_term / self.img_set_size + second_term / self.txt_set_size) / self.denominator

        return chamfer_dist
    
    def max_similarity_cosine(self, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs)
        max_similarity = self.xy_max_pool(dist.unsqueeze(0)).squeeze(0)
        return max_similarity

    def smooth_chamfer_similarity(self, img_embs, txt_embs):
        return self.smooth_chamfer_similarity_cosine(img_embs, txt_embs)
    
    def chamfer_similarity(self, img_embs, txt_embs):
        return self.chamfer_similarity_cosine(img_embs, txt_embs)
    
    def max_similarity(self, img_embs, txt_embs):
        return self.max_similarity_cosine(img_embs, txt_embs)
    
    def avg_similarity(self, img_embs, txt_embs):
        return self.mp_dist(img_embs, txt_embs)