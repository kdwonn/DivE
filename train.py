import os 
import math
import shutil
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.cuda.amp
import numpy as np
import wandb

import data
from vocab import Vocabulary
from loss import AsymmetricTripletLoss
from eval import i2t, t2i, encode_data
from logger import AverageMeter
from option import parser, verify_input_args
from sync_batchnorm import convert_model, SynchronizedBatchNorm2d
from similarity import SetwiseDistance
from model_spm import VSE

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
total_iter = 0

def save_ckpt(state, is_best, filename='ckpt.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        logging.info('Updating the best model checkpoint: {}'.format(prefix + 'model_best.pth.tar'))


def get_description(args, epoch=-1):
    return ('[{}][epoch:{}] {}'.format(args.logger_name.split('/')[-1], epoch, args))


def train(epoch, data_loader, model, criterion, optimizer, scaler, args, lr_warmup, scheduler=None):
    global total_iter
    # switch to train mode
    model.train()
    if args.bn_eval:
        modules = model.module.modules() if args.multi_gpu else model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
    
    # debug_criterion = ChamferLoss(args, smooth=True)
    # average meters to record the training statistics
    losses = AverageMeter()
    stat_dict = dict()
    losses_dict = dict()
    # losses_dict['ranking_loss'] = AverageMeter()
    losses_dict['i2t_loss'] = AverageMeter()
    losses_dict['t2i_loss'] = AverageMeter()
    if args.mmd_weight > 0:
        losses_dict['mmd_loss'] = AverageMeter()
    if args.unif_weight > 0:
        losses_dict['unif_loss'] = AverageMeter()

    total_batches = len(data_loader)
    
    for itr, data in enumerate(data_loader):
        total_iter += 1
        if torch.cuda.is_available():
            if 'butd' in args.data_name:
                img, txt, img_len, txt_len, recovery, _ = data
                img, txt, img_len, txt_len, recovery = img.cuda(), txt.cuda(), img_len.cuda(), txt_len.cuda(), recovery.cuda()
            else:
                img_len = None
                img, txt, txt_len, recovery, _ = data
                img, txt, txt_len, recovery = img.cuda(), txt.cuda(), txt_len.cuda(), recovery.cuda()
        else: 
            assert False
            
        with torch.cuda.amp.autocast(enabled=args.amp):
            # Forward pass and compute loss; _a: attention map, _r: residuals
            img_emb, txt_emb, img_sel_mask, txt_sel_mask, img_r, txt_r = model.forward(img, txt, img_len, txt_len)
            # Compute loss and update statstics. Give loss a recovery label when args.fast_batch is on.
            txt_emb = txt_emb[recovery]
            txt_r = txt_r[recovery]
            
            loss, loss_dict = criterion(img_emb, txt_emb, img_r, txt_r)

            # if total_iter < lr_warmup:
            #     loss *= float(total_iter) / lr_warmup
            
            if lr_warmup:
                loss *= float(itr) /  len(data_loader)
        
        if torch.isnan(loss).any():
            print("!! NaN loss detected !!")
            import ipdb; ipdb.set_trace()
            
        losses.update(loss)
        for key, val in loss_dict.items():
            losses_dict[key].update(val)

        # Backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        wandb.log({'iter':total_iter})
        if scheduler is not None and total_iter >= lr_warmup:
            scheduler.step()
        
        # Print log info
    log_msg = 'loss: %.4f (%.4f)' %(losses.val, losses.avg)
    for key, val in losses_dict.items():
        log_msg += ', %s: %.4f, (%.4f)' %(key.replace('_loss',''), val.val, val.avg)
    logging.info('[%d] %s' %(epoch, log_msg))

    del img_emb, txt_emb, img_r, txt_r, loss
    return losses.avg, losses_dict, stat_dict
        

def validate(dataset, data_loader, model, args, distance_fn, validation, epoch=-1, best_score=None):
    # switch to eval mode
    model.eval()

    nreps = 5 if args.data_name in ['f30k', 'coco', 'coco_butd', 'f30k_butd'] else 10

    img_embs, txt_embs = encode_data(model, data_loader, 'butd' in args.data_name, args.eval_on_gpu)
    # 5fold cross-validation, only for MSCOCO
    mean_metrics = None
    if 'coco' in args.data_name and not validation:
        results = []
        for i in range(5):
            r, rt0 = i2t(
                img_embs[i*5000:(i + 1)*5000], txt_embs[i*5000:(i + 1)*5000], 
                distance_fn,
                nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
            
            ri, rti0 = t2i(
                img_embs[i*5000:(i + 1)*5000], txt_embs[i*5000:(i + 1)*5000], 
                distance_fn,
                nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
            
            r = (r[0], r[1], r[2], r[3], r[3] / img_embs.shape[0], r[4], r[4] / img_embs.shape[0])
            # print("Image to text: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % r)
            ri = (ri[0], ri[1], ri[2], ri[3], ri[3] / img_embs.shape[0], ri[4], ri[4] / img_embs.shape[0])
            # print("Text to image: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % ri)

            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            # print("rsum: %.2f ar: %.2f ari: %.2f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())

        print("-----------------------------------")
        print("Mean metrics from 5-fold evaluation: ")
        print("rsum: %.2f" % (mean_metrics[-1]))
        print("Average i2t Recall: %.2f" % mean_metrics[-3])
        print("Image to text: %.2f %.2f %.2f" % mean_metrics[:3])
        print("Average t2i Recall: %.2f" % mean_metrics[-2])
        print("Text to image: %.2f %.2f %.2f" % mean_metrics[7:10])
    
        recall_1k = (mean_metrics[0], mean_metrics[1], mean_metrics[2], mean_metrics[7], mean_metrics[8], mean_metrics[9])
    else:
        recall_1k = (0, 0, 0, 0, 0, 0)
    
    (r1, r5, r10, medr, meanr), (ranks, top1) = i2t(img_embs, txt_embs, distance_fn,
            nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
    (r1i, r5i, r10i, medri, meanri), (ranksi, top1i) = t2i(img_embs, txt_embs, distance_fn,
            nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)

    # sum of recalls to be used for early stopping
    rsum = r1 + r5 + r10 + r1i + r5i + r10i
    med_rsum, mean_rsum = medr + medri, meanr + meanri

    # log
    exp_name = args.logger_name.split('/')[-1]
    vname = 'Image'

    log_str1 = "[%s][%d] %s to text: %.2f, %.2f, %.2f, %.2f, %.2f" \
                            %(exp_name, epoch, vname, r1, r5, r10, medr, meanr)
    log_str2 = "[%s][%d] Text to %s: %.2f, %.2f, %.2f, %.2f, %.2f" \
                            %(exp_name, epoch, vname, r1i, r5i, r10i, medri, meanri)
    log_str3 = '[%s][%d] rsum: %.2f, med_rsum: %.2f, mean_rsum: %.2f' \
                            %(exp_name, epoch, rsum, med_rsum, mean_rsum)
    if best_score:
        log_str3 += ' (best %s: %.2f)' %(args.val_metric, best_score)

    i2t_recall, t2i_recall = (r1, r5, r10), (r1i, r5i, r10i) 
    
    logging.info(log_str1)
    logging.info(log_str2)
    logging.info(log_str3)

    dscr = get_description(args, epoch)
    log_msg = '{}\n{}\n{}'.format(log_str1, log_str2, log_str3)

    if args.val_metric == 'rsum':
        return rsum, i2t_recall, t2i_recall, recall_1k
    elif args.val_metric == 'med_rsum':
        return med_rsum, i2t_recall, t2i_recall, recall_1k
    else:
        return mean_rsum, i2t_recall, t2i_recall, recall_1k


def update_best_score(new_score, old_score, is_higher_better):
    if not old_score:
        score, updated = new_score, True
    else:
        if is_higher_better:
            score = max(new_score, old_score)
            updated = new_score > old_score
        else:
            score = min(new_score, old_score)
            updated = new_score < old_score
    return score, updated

def warmup(model, epoch, args, multi_gpu):
    if args.img_finetune and args.txt_finetune:
        warm = epoch >= args.warm_epoch
        if args.warm_img:
            for idx, param in enumerate((model.module if multi_gpu else model).img_enc.cnn.parameters()):
                param.requires_grad = warm
        if args.warm_txt:
            (model.module if multi_gpu else model).txt_enc.embed.weight.requires_grad = warm

def finetune_lr_lower(optimizer, epoch, args):
    if epoch == args.warm_epoch:
        for g in optimizer.param_groups:
            g['lr'] *= args.finetune_lr_lower

def tri_mean_to_max(criterion, epoch, args):
    if args.tri_mean_to_max:
        criterion.max_violation = epoch >= args.warm_epoch     
        
def main():
    args = verify_input_args(parser.parse_args())
    print(args)
    
    LOG_DIR = os.path.join(args.log_dir, args.remark)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    args.log_dir = LOG_DIR
    args.logger_name = LOG_DIR
    wandb.init(project='cross_modal_retrieval', notes=args.log_dir, name = args.remark)
    wandb.config.update(args)

    # Load Vocabulary Wrapper
    vocab_path = os.path.join(args.vocab_path, '%s_vocab.pkl' % args.data_name)
    vocab = pickle.load(open(vocab_path, 'rb'))
    vocab.add_word('<mask>')
    print('Add <mask> token into the vocab')

    # Dataloaders
    if args.data_name in ['coco', 'f30k', 'coco_butd', 'f30k_butd']:
        txt_per_img = 5 
    else:
        raise NotImplementedError

    if args.data_name in ['coco', 'f30k', 'coco_butd', 'f30k_butd']:
        trn_loader, val_loader = data.get_loaders(args, vocab)
    else:
        raise NotImplementedError

    # Construct the model
    if 'butd' in args.data_name:
        model = VSE(vocab.word2idx, args)
    else:
        model = VSE(vocab.word2idx, args)
            
    if torch.cuda.is_available():
        if args.multi_gpu:
            model = nn.DataParallel(model, output_device=1)
        if args.sync_bn:
            model = convert_model(model)
        model = model.cuda()
        cudnn.benchmark = True
        
    wandb.watch(models=model, log_freq=1000, log='gradients')
    
    # distance function options
    train_distance = SetwiseDistance(args.img_num_embeds, args.txt_num_embeds, args.denominator, args.temperature, args.temperature_txt_scale)
    if args.loss == 'smooth_chamfer':
        train_distance_fn = train_distance.smooth_chamfer_distance
    elif args.loss == 'chamfer':
        train_distance_fn = train_distance.chamfer_distance
    elif args.loss == 'max':
        train_distance_fn = train_distance.max_distance
    elif args.loss == 'mp':
        train_distance_fn = train_distance.avg_distance
    else:
        assert False
    
    eval_distance = SetwiseDistance(args.img_num_embeds, args.txt_num_embeds, \
        args.denominator, args.temperature, args.temperature_txt_scale)
    if args.eval_distance == 'smooth_chamfer':
        eval_distance_fn = eval_distance.smooth_chamfer_distance
    elif args.eval_distance == 'chamfer':
        eval_distance_fn = eval_distance.chamfer_distance
    elif args.eval_distance == 'max':
        eval_distance_fn = eval_distance.max_distance
    elif args.loss == 'mp':
        eval_distance_fn = train_distance_fn
    else:
        assert False
            
    # Loss and optimizer
    if args.loss in ['smooth_chamfer', 'chamfer', 'max', 'mp']:
        # assert args.fast_batch
        criterion = AsymmetricTripletLoss(
            img_set_size=args.img_num_embeds, 
            txt_set_size=args.txt_num_embeds, 
            distance_fn=train_distance_fn, 
            opt=args, txt_per_img=txt_per_img
        )
    else:
        assert False
        
    module = model.module if args.multi_gpu else model
    param_groups = [
        {'params': list(set(module.img_enc.parameters()).difference(set(module.img_enc.cnn.parameters()))), 'lr': args.lr * args.img_pie_lr_scale},
        {'params': module.img_enc.cnn.parameters(), 'lr': args.lr},
        {'params': list(set(module.txt_enc.parameters()).difference(set(module.txt_enc.spm.parameters()))), 'lr': args.lr * args.txt_lr_scale},
        {'params': list(set(module.txt_enc.spm.parameters())), 'lr': args.lr * args.txt_pie_lr_scale}
    ]
    if args.loss == 'mp':
        param_groups += [{'params':train_distance.parameters(), 'lr': args.lr}]
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamp':
        from adamp import AdamP
        optimizer = AdamP(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trn_loader)*args.num_epochs)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma = args.lr_step_gamma)
    
    best_score = 0
    
    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    for epoch in range(args.num_epochs):
        #warm up training data
        warmup(model, epoch, args, args.multi_gpu)
        finetune_lr_lower(optimizer, epoch, args)
        tri_mean_to_max(criterion, epoch, args)

        if args.lr_scheduler == 'pvse_cosine' and epoch == args.warm_epoch:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warm_epoch, T_mult=1)
        
        warm_iter = epoch == args.lr_warmup
        loss, losses_dict, stat_dict = train(
            epoch, trn_loader, model, criterion, optimizer, scaler, args, warm_iter, 
            scheduler=lr_scheduler if args.lr_scheduler == 'cosine' else None
        )
        
        wandb.log({"epoch": epoch}, step=total_iter)
        wandb.log({"Loss": loss}, step=total_iter)
        for key, val in losses_dict.items():
            wandb.log({key: val.avg}, step=total_iter)
        for key, val in stat_dict.items():
            wandb.log({key: val.avg}, step=total_iter)
        wandb.log({"LR" : optimizer.param_groups[0]['lr']}, step=total_iter)
        
        # evaluate on validation set
        with torch.no_grad():
            if epoch % args.eval_epoch == 0:
                val_score, i2t_recall, t2i_recall, recall_1k = validate(None, val_loader, model, args, eval_distance_fn, True, epoch, best_score)
                wandb.log({"val i2t R@1" : i2t_recall[0]}, step=total_iter)
                wandb.log({"val i2t R@5" : i2t_recall[1]}, step=total_iter)
                wandb.log({"val i2t R@10" : i2t_recall[2]}, step=total_iter)

                wandb.log({"val t2i R@1" : t2i_recall[0]}, step=total_iter)
                wandb.log({"val t2i R@5" : t2i_recall[1]}, step=total_iter)
                wandb.log({"val t2i R@10" : t2i_recall[2]}, step=total_iter)
                
                rsum_val = i2t_recall[0]+i2t_recall[1]+i2t_recall[2]+t2i_recall[0]+t2i_recall[1]+t2i_recall[2]
                wandb.log({"val rsum": rsum_val}, step=total_iter)

                # remember best rsum and save ckpt
                best_score, updated = update_best_score(rsum_val, best_score, args.val_metric=='rsum')
                save_ckpt({
                    'args': args,
                    'epoch': epoch,
                    'best_score': best_score,
                    'model': model.state_dict(),
                }, updated, prefix=args.logger_name + '/')
        
        # adjust learning rate if rsum stagnates
        if args.lr_scheduler != 'cosine':
            lr_scheduler.step()

if __name__ == '__main__':
    main()
