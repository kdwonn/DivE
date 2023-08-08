from __future__ import print_function
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from vocab import Vocabulary
from model_pie import PVSE
from model_spm import VSE
from data import get_test_loader, get_loaders
from option import parser, verify_input_args
from similarity import SetwiseSimilarity

def encode_data(model, data_loader, butd, use_gpu=False):
    """Encode all images and sentences loadable by data_loader"""
    # switch to evaluate mode
    model.eval()

    use_mil = model.module.mil if hasattr(model, 'module') else model.mil

    # numpy array to keep all the embeddings
    img_embs, txt_embs = None, None
    for i, data in tqdm(enumerate(data_loader)):
        if butd:
            img, txt, img_len, txt_len, ids = data
            img, txt, img_len, txt_len = img.cuda(), txt.cuda(), img_len.cuda(), txt_len.cuda()
        else:
            img_len = None
            img, txt, txt_len, ids = data
            img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

        # compute the embeddings
        img_emb, txt_emb, _, _, _, _ = model.forward(img, txt, img_len, txt_len)
        del img, txt, img_len, txt_len

        # initialize the output embeddings
        if img_embs is None:
            img_emb_sz = [len(data_loader.dataset), img_emb.size(1), img_emb.size(2)] \
                    if use_mil else [len(data_loader.dataset), img_emb.size(1)]
            txt_emb_sz = [len(data_loader.dataset), txt_emb.size(1), txt_emb.size(2)] \
                    if use_mil else [len(data_loader.dataset), txt_emb.size(1)]
            img_embs = torch.zeros(img_emb_sz, dtype=img_emb.dtype, requires_grad=False).cuda()
            txt_embs = torch.zeros(txt_emb_sz, dtype=txt_emb.dtype, requires_grad=False).cuda()

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb 
        txt_embs[ids] = txt_emb 

    return img_embs, txt_embs


def i2t(images, sentences, similarity_fn, nreps=1, npts=None, return_ranks=False, use_gpu=False, is_cosine=True):
    """
    Images->Text (Image Annotation)
    Images: (nreps*N, K) matrix of images
    Captions: (nreps*N, K) matrix of sentences
    """
    # NOTE nreps : numbrt of captions per image, npts: number of images
    if npts is None:
        npts = int(images.shape[0] / nreps)
        
    index_list = []
    ranks, top1 = np.zeros(npts), np.zeros(npts)
    for index in range(npts):
        # Get query image
        im = images[nreps * index]
        im = im.reshape((1,) + im.shape)
        # Compute scores
        if use_gpu:
            if len(sentences.shape) == 2:
                sim = im.mm(sentences.t()).view(-1)
            else:
                _, K, D = im.shape
                sim = similarity_fn(im.view(-1, D), sentences.view(-1, D)).flatten()
        else: 
            sim = np.tensordot(im, sentences, axes=[2, 2]).max(axis=(0,1,3)).flatten() \
                    if len(sentences.shape) == 3 else np.dot(im, sentences.T).flatten()

        if use_gpu:
            _, inds_gpu = sim.sort()
            inds = inds_gpu.cpu().numpy().copy()[::-1] #reverse order / change it to descending order
        else:
            inds = np.argsort(sim)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(nreps * index, nreps * (index + 1), 1):
            tmp = np.where(inds == i)[0][0] # find the rank of given text data
            if tmp < rank:
                rank = tmp
            # find highest rank among matching queries
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    # import ipdb; ipdb.set_trace()
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, sentences, similarity_fn, nreps=1, npts=None, return_ranks=False, use_gpu=False, is_cosine=True):
    """
    Text->Images (Image Search)
    Images: (nreps*N, K) matrix of images
    Captions: (nreps*N, K) matrix of sentences
    """
    if npts is None:
        npts = int(images.shape[0] / nreps)

    if use_gpu:
        ims = torch.stack([images[i] for i in range(0, len(images), nreps)])
    else:
        ims = np.array([images[i] for i in range(0, len(images), nreps)])

    ranks, top1 = np.zeros(nreps * npts), np.zeros(nreps * npts)
    for index in range(npts):
        # Get query sentences
        queries = sentences[nreps * index:nreps * (index + 1)]

        # Compute scores
        if use_gpu:
            if len(sentences.shape) == 2:
                sim = queries.mm(ims.t())
            else:
                sim = similarity_fn(ims.view(-1, ims.size(-1)), queries.view(-1, queries.size(-1))).t()
        else:
            sim = np.tensordot(queries, ims, axes=[2, 2]).max(axis=(1,3)) \
                    if len(sentences.shape) == 3 else np.dot(queries, ims.T)

        inds = np.zeros(sim.shape)
        for i in range(len(inds)):
            if use_gpu:
                _, inds_gpu = sim[i].sort()
                inds[i] = inds_gpu.cpu().numpy().copy()[::-1]
            else:
                inds[i] = np.argsort(sim[i])[::-1]
            ranks[nreps * index + i] = np.where(inds[i] == index)[0][0]
            top1[nreps * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)



def convert_old_state_dict(x, model, multi_gpu=False):
    params = model.state_dict()
    prefix = ['module.img_enc.', 'module.txt_enc.'] \
            if multi_gpu else ['img_enc.', 'txt_enc.']
    for i, old_params in enumerate(x):
        for key, val in old_params.items():
            key = prefix[i] + key.replace('module.','').replace('our_model', 'pie_net')
            assert key in params, '{} not found in model state_dict'.format(key)
            params[key] = val
    return params



def evalrank(model, args, split='test'):
    print('Loading dataset')
    if args.data_name in ['f30k', 'coco', 'coco_butd', 'f30k_butd']:
        if split == 'val':
            _, data_loader = get_loaders(args, vocab)
            dataset = data_loader.dataset
        elif split == 'test':
            dataset, data_loader = None, get_test_loader(args, vocab)

    print('Computing results... (eval_on_gpu={})'.format(args.eval_on_gpu))
    img_embs, txt_embs = encode_data(model, data_loader, 'butd' in args.data_name, args.eval_on_gpu)
    n_samples = img_embs.shape[0]

    nreps = 5
    print('Images: %d, Sentences: %d' % (img_embs.shape[0] / nreps, txt_embs.shape[0]))
    
    img_set_size, txt_set_size = args.img_num_embeds, args.txt_num_embeds
    similarity = SetwiseSimilarity(
        img_set_size, txt_set_size, args.denominator, args.temperature)
    if args.loss == 'smooth_chamfer':
        similarity_fn = similarity.smooth_chamfer_similarity
    elif args.loss == 'chamfer':
        similarity_fn = similarity.chamfer_similarity
    elif args.loss == 'max':
        similarity_fn = similarity.max_similarity
    else:
        raise NotImplementedError
    
    # 5fold cross-validation, only for MSCOCO
    mean_metrics = None
    if 'coco' in args.data_name and split == 'test':
        results = []
        for i in range(5):
            r, rt0 = i2t(
                img_embs[i*5000:(i + 1)*5000], txt_embs[i*5000:(i + 1)*5000], 
                similarity_fn,
                nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
            
            ri, rti0 = t2i(
                img_embs[i*5000:(i + 1)*5000], txt_embs[i*5000:(i + 1)*5000], 
                similarity_fn,
                nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
            
            r = (r[0], r[1], r[2], r[3], r[3] / n_samples, r[4], r[4] / n_samples)
            print("Image to text: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % r)
            ri = (ri[0], ri[1], ri[2], ri[3], ri[3] / n_samples, ri[4], ri[4] / n_samples)
            print("Text to image: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % ri)

            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.2f ar: %.2f ari: %.2f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())

        print("-----------------------------------")
        print("Mean metrics from 5-fold evaluation: ")
        print("rsum: %.2f" % (mean_metrics[-1]))
        print("Average i2t Recall: %.2f" % mean_metrics[-3])
        print("Image to text: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % mean_metrics[:7])
        print("Average t2i Recall: %.2f" % mean_metrics[-2])
        print("Text to image: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % mean_metrics[7:14])

    # no cross-validation, full evaluation
    r, rt = i2t(img_embs, txt_embs, similarity_fn, nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
    ri, rti = t2i(img_embs, txt_embs, similarity_fn, nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
        
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    r = (r[0], r[1], r[2], r[3], r[3] / n_samples, r[4], r[4] / n_samples)
    ri = (ri[0], ri[1], ri[2], ri[3], ri[3] / n_samples, ri[4], ri[4] / n_samples)
    print("rsum: %.2f" % rsum)
    print("Average i2t Recall: %.2f" % ar)
    print("Image to text: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % r)
    print("Average t2i Recall: %.2f" % ari)
    print("Text to image: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % ri)

    return mean_metrics


if __name__ == '__main__':
    args = verify_input_args(parser.parse_args())
    opt = verify_input_args(parser.parse_args())

    # load vocabulary used by the model
    with open('./vocab/%s_vocab.pkl' % args.data_name, 'rb') as f:
        vocab = pickle.load(f)
    args.vocab_size = len(vocab)
    vocab.add_word('<mask>')
    print('Add <mask> token into the vocab')

    # load model and options
    if args.ckpt == '':
        args.ckpt = os.path.join('./logs', args.remark, 'ckpt.pth.tar')
        print(args.ckpt)
    assert os.path.isfile(args.ckpt)
    
    if args.arch == 'pvse':
        model = PVSE(vocab.word2idx, args)
    elif args.arch == 'slot':
        model = VSE(vocab.word2idx, args)
        
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda() if args.multi_gpu else model.cuda()
        torch.backends.cudnn.benchmark = True
        
    state_dict = torch.load(args.ckpt)['model']
    
    model.load_state_dict(state_dict)
    with torch.no_grad():
        # evaluate
        metrics = evalrank(model, args, split='val')
        metrics = evalrank(model, args, split='test')
