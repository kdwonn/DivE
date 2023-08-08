import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence 
from torch.distributions import Normal
from similarity import SetwiseSimilarity
from einops import rearrange, repeat

def cosine_sim(x, y):
    """Cosine similarity between all the image and sentence pairs. Assumes x and y are l2 normalized"""
    return x.mm(y.t())

def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)

def rbf(x, y, gamma):
    """RBF kernel K(x,y) """
    pdist = torch.norm(x[:, None] - y, dim=2, p=2)
    return torch.exp(-gamma * pdist)

def rbf_memory_efficient(x, y, gamma):
    """RBF kernel that does not cause memory shortage"""
    cdist = torch.cdist(x, y)
    return torch.exp(-gamma * cdist)

class TripletLoss(nn.Module):
    def __init__(self, img_set_size, txt_set_size, similarity_fn, opt, reduction='mean', txt_per_img=5):
        super(TripletLoss, self).__init__()
        
        # loss hyperparameters
        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.img_num_embeds = opt.img_num_embeds
        self.txt_num_embeds = opt.txt_num_embeds
        self.reduction = reduction
        self.img_emb_norm = []
        self.txt_emb_norm = []
        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.mmd_weight = opt.mmd_weight if hasattr(opt, 'mmd_weight') else 0.
        self.div_weight = opt.div_weight if hasattr(opt, 'div_weight') else 0.
        self.unif_weight = opt.unif_weight if hasattr(opt, 'unif_weight') else 0.
        self.qreg_weight = opt.qreg_weight if hasattr(opt, 'qreg_weight') else 0.
        self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
        self.unif_residual = opt.unif_residual
        
        # set_similarity hyperparameters
        self.similarity_fn = similarity_fn
        self.img_set_size, self.txt_set_size = img_set_size, txt_set_size
        self.txt_per_img = txt_per_img
        """
        set_per_img : Matching sets per each image. 2 scenarios.
        Lets assumes that 
            K : number of embeddings for each sample, 
            T : number of matching captions per image (5 in COCO).
        set_per_img = (T * K) / txt_set_size
        1. set_per_img = T, txt_set_size = K
        2. set_per_img = 1, txt_set_size = T * K
        """
        self.set_per_img = int(self.txt_per_img * self.txt_num_embeds / self.txt_set_size) 
        self.semi_hard_triplet = opt.semi_hard_triplet
        
    def diversity_loss(self, x, num_embeds):
        if num_embeds == 1:
            return 0.0
        x = l2norm(x) # Columns of x MUST be l2-normalized
        gram_x = x.bmm(x.transpose(1,2))
        I = torch.tensor((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1))
        if torch.cuda.is_available():
            I = I.cuda()
        gram_x.masked_fill_(I, 0.0)
        loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (num_embeds**2)
        return loss.mean() if self.reduction=='mean' else loss.sum()

    def mmd_rbf_loss(self, x, y, gamma=None):
        if gamma is None:
            gamma = 1./x.size(-1)
        if self.reduction=='mean':
            loss = rbf_memory_efficient(x, x, gamma).mean() - 2 * rbf_memory_efficient(x, y, gamma).mean() + rbf_memory_efficient(y, y, gamma).mean()
        else:
            loss = rbf_memory_efficient(x, x, gamma).sum() - 2 * rbf_memory_efficient(x, y, gamma).sum() + rbf_memory_efficient(y, y, gamma).sum()
        return loss
    
    def batchwise_uniformity_loss(self, embs, num_embeds, t=2):
        if num_embeds == 1:
            return 0.0
        rbf = torch.exp(-t * torch.cdist(embs, embs).pow(2))
        I = torch.tensor(repeat(
            torch.triu(torch.ones(rbf.shape[1], rbf.shape[1]), diagonal=1), 
            'n d -> b n d', 
            b=rbf.shape[0]
        )).cuda()
        rbf = torch.where(I == 1, rbf, torch.zeros_like(rbf))
        loss = torch.stack([r.sum() for r in rbf]) / (num_embeds * (num_embeds - 1) * 0.5)
        return loss.mean()
    
    def query_regularizer(self, img_query, txt_query, t=2):
        return torch.exp(-t * F.pdist(img_query)).mean() + torch.exp(-t * F.pdist(txt_query)).mean()
    
    def triplet_ranking_loss(self, A, B, max_dim):
        if self.semi_hard_triplet:
            loss = (self.margin + A - B).clamp(min=0.0, max=self.margin)
            num_triplets = torch.nonzero(loss).shape[0]
            if num_triplets == 0:
                return loss.mean()
            else:
                return loss.sum() / num_triplets
        else:
            loss = (self.margin + A - B).clamp(min=0.0)
            if self.max_violation:
                loss = loss.max(max_dim)[0]
            return loss.mean()
    
    def forward(self, img_embs, txt_embs, img_r, txt_r, img_query=None, txt_query=None):
        loss, losses = 0, dict()
        
        # compare every diagonal score to scores in its column (image-to-text retrieval)
        self.img_emb_norm += [img_embs.reshape(-1, img_embs.shape[-1]).norm(dim=1).mean().item()]
        self.txt_emb_norm += [txt_embs.reshape(-1, txt_embs.shape[-1]).norm(dim=1).mean().item()]
        
        # reshape embeddings as 2D tensors (given as 3D tensors).
        img_embs = img_embs.reshape(-1, img_embs.shape[-1])
        txt_embs = txt_embs.reshape(-1, txt_embs.shape[-1])
        
        # Compute setwise similarity with provided set similarity metric
        setwise_dist = self.similarity_fn(img_embs, txt_embs)
        
        # generate mask based on the computed number of sets per images
        mask = (torch.eye(setwise_dist.shape[0]) > .5).cuda()
        mask = mask.view(-1, 1).repeat(1, self.set_per_img).reshape(setwise_dist.shape)
        
        """
        example when txt_set_size : 5, set_per_img : 5>
            coco dataset, caption per image : 5 
            setwise_dist : (128, 640)
            mask : (128, 640) , indicates matching pairs
            i2t_pos : (5, 128, 1) 
            i2t_neg : (5, 128, 635) , repeat of (1, 128, 635)
        """
        
        neg_mask = ~mask
        # i2t loss. multiple matching captions exist for each each image
        i2t_pos = setwise_dist[mask].view(setwise_dist.shape[0], -1, 1).permute(1, 0, 2)
        i2t_neg = setwise_dist[neg_mask].view(1, setwise_dist.shape[0], -1)
        i2t_loss = self.triplet_ranking_loss(i2t_neg, i2t_pos, 2)
        
        # t2i loss. single matching image exists for each each caption
        t2i_pos = setwise_dist.t()[mask.t()].reshape(setwise_dist.shape[1], -1)
        t2i_neg = setwise_dist.t()[neg_mask.t()].reshape(setwise_dist.shape[1], -1)
        t2i_loss = self.triplet_ranking_loss(t2i_neg, t2i_pos, 1)
        
        losses['t2i_loss'] = t2i_loss
        losses['i2t_loss'] = i2t_loss 
        loss += i2t_loss + t2i_loss
        
        if self.div_weight > 0.:
            div_loss = self.diversity_loss(img_r, self.img_num_embeds) + \
                self.diversity_loss(txt_r, self.txt_num_embeds)
            loss += self.div_weight * div_loss
            losses['div_loss'] = div_loss
        
        # domain discrepancy loss
        if self.mmd_weight > 0.:
            mmd_loss = self.mmd_rbf_loss(img_embs, txt_embs, gamma=0.5)
            loss += self.mmd_weight * mmd_loss
            losses['mmd_loss'] = mmd_loss
            
        if self.unif_weight > 0.:
            unif_img = l2norm(img_r) if self.unif_residual else img_embs
            unif_txt = l2norm(txt_r) if self.unif_residual else txt_embs
            unif_loss = self.batchwise_uniformity_loss(unif_img.reshape(-1, self.img_num_embeds, unif_img.shape[-1]), self.img_num_embeds) + \
                self.batchwise_uniformity_loss(unif_txt.reshape(-1, self.txt_num_embeds, unif_txt.shape[-1]), self.txt_num_embeds)
            loss += self.unif_weight * unif_loss
            losses['unif_loss'] = unif_loss
            
        if self.qreg_weight > 0. and (img_query is not None):
            qreg = self.query_regularizer(img_query, txt_query)
            loss += self.qreg_weight * qreg
            losses['query_regularizer'] = qreg

        return loss, losses