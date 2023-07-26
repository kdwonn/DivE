import os
import sys
import random

import numpy as np
import json as jsonmod

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import nltk
from PIL import Image
from pycocotools.coco import COCO
from transformers import BertTokenizer


def get_paths(path, name='coco', use_restval=True):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...   A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the `restval` data is included in train for COCO dataset.
    """
    roots, ids = {}, {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json'),
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json'),
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json'),
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap']),
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (ids['train'],
                np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif name in ['coco_butd', 'f30k_butd']:
        imgdir = os.path.join(path, 'precomp')
        cap = os.path.join(path, 'precomp')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


def tokenize(sentence, vocab, drop_prob):
    # Convert sentence (string) to word ids.
    def caption_augmentation(tokens):
        idxs = []
        for t in tokens:
            prob = random.random()
            if prob < drop_prob:
                prob /= drop_prob
                if prob < 0.5:
                    idxs += [vocab('<mask>')]
                elif prob < 0.6:
                    idxs += [random.randrange(len(vocab))]
            else:
                idxs += [vocab(t)]
        return idxs
    
    if sys.version_info.major > 2:
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
    else:
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower().decode('utf-8'))
    return torch.Tensor(
        [vocab('<start>')] + caption_augmentation(tokens) + [vocab('<end>')]
    )


class CocoDataset(data.Dataset):

    def __init__(self, root, json, vocab, split, transform=None, ids=None, drop_prob=0):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        
        # if ids provided by get_paths, use split-specific ids
        self.ann_ids = list(self.coco.anns.keys()) if ids is None else ids
        if not isinstance(self.ann_ids, tuple):
            self.ann_ids = (self.ann_ids, [])
        
        self.vocab = vocab
        self.transform = transform
        self.drop_prob = drop_prob

        # if `restval` data is to be used, record the break point for ids
        self.ann_bp = len(self.ann_ids[0])
        self.ann_ids = list(self.ann_ids[0]) + list(self.ann_ids[1])
        
        from collections import defaultdict
        self.img_id_to_ann_ids = (defaultdict(list), defaultdict(list))
        for i, ann_id in enumerate(self.ann_ids):
            is_beyond_bp = int(i >= self.ann_bp)
            coco, root, img_id_to_ann_ids =\
                self.coco[is_beyond_bp], self.root[is_beyond_bp], self.img_id_to_ann_ids[is_beyond_bp]
            img_id = coco.anns[ann_id]['image_id']
            img_id_to_ann_ids[img_id].append(ann_id)
            
        self.img_ids = (list(self.img_id_to_ann_ids[0].keys()), list(self.img_id_to_ann_ids[1].keys()))
        self.img_bp = len(self.img_ids[0])
        self.img_ids = self.img_ids[0] + self.img_ids[1]
        
        print(self.img_bp, self.ann_bp, len(self.img_ids), len(self.ann_ids))

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        vocab = self.vocab
        ann_ids, anns, path, image = self.get_raw_item(index)
        if self.transform is not None:
            image = self.transform(image)
        anns = [tokenize(ann, vocab, self.drop_prob) for ann in anns]
        
        return image, anns, index, ann_ids


    def get_raw_item(self, index):
        is_beyond_bp = int(index >= self.img_bp)
        coco, root, img_id_to_ann_ids = \
            self.coco[is_beyond_bp], self.root[is_beyond_bp], self.img_id_to_ann_ids[is_beyond_bp]
        
        img_id = self.img_ids[index]
        ann_ids = img_id_to_ann_ids[img_id]
        assert len(ann_ids) == 5
        anns = [coco.anns[i]['caption'] for i in ann_ids]
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        
        return ann_ids, anns, path, image


class CocoDatasetBert(data.Dataset):

    def __init__(self, root, json, vocab, split, transform=None, ids=None, drop_prob=0):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        self.train = split == 'train'
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        
        # if ids provided by get_paths, use split-specific ids
        self.ann_ids = list(self.coco.anns.keys()) if ids is None else ids
        if not isinstance(self.ann_ids, tuple):
            self.ann_ids = (self.ann_ids, [])
        
        self.transform = transform
        self.drop_prob = drop_prob
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab

        # if `restval` data is to be used, record the break point for ids
        self.ann_bp = len(self.ann_ids[0])
        self.ann_ids = list(self.ann_ids[0]) + list(self.ann_ids[1])
        
        from collections import defaultdict
        self.img_id_to_ann_ids = (defaultdict(list), defaultdict(list))
        for i, ann_id in enumerate(self.ann_ids):
            is_beyond_bp = int(i >= self.ann_bp)
            coco, root, img_id_to_ann_ids =\
                self.coco[is_beyond_bp], self.root[is_beyond_bp], self.img_id_to_ann_ids[is_beyond_bp]
            img_id = coco.anns[ann_id]['image_id']
            img_id_to_ann_ids[img_id].append(ann_id)
            
        self.img_ids = (list(self.img_id_to_ann_ids[0].keys()), list(self.img_id_to_ann_ids[1].keys()))
        self.img_bp = len(self.img_ids[0])
        self.img_ids = self.img_ids[0] + self.img_ids[1]
        
        print(self.img_bp, self.ann_bp, len(self.img_ids), len(self.ann_ids))

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        vocab = self.vocab
        ann_ids, anns, path, image = self.get_raw_item(index)
        if self.transform is not None:
            image = self.transform(image)
        anns = [process_caption_bert(ann, self.tokenizer, self.drop_prob, self.train) for ann in anns]
        
        return image, anns, index, ann_ids


    def get_raw_item(self, index):
        is_beyond_bp = int(index >= self.img_bp)
        coco, root, img_id_to_ann_ids = \
            self.coco[is_beyond_bp], self.root[is_beyond_bp], self.img_id_to_ann_ids[is_beyond_bp]
        
        img_id = self.img_ids[index]
        ann_ids = img_id_to_ann_ids[img_id]
        assert len(ann_ids) == 5
        anns = [coco.anns[i]['caption'] for i in ann_ids]
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        
        return ann_ids, anns, path, image


class CocoDatasetTest(data.Dataset):

    def __init__(self, root, json, vocab, split, transform=None, ids=None, drop_prob=0):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)

        # if ids provided by get_paths, use split-specific ids
        self.ids = list(self.coco.anns.keys()) if ids is None else ids
        self.vocab = vocab
        self.transform = transform
        self.drop_prob = drop_prob

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        vocab = self.vocab
        root, sentence, img_id, path, image = self.get_raw_item(index)
        if self.transform is not None:
            image = self.transform(image)

        target = tokenize(sentence, vocab, self.drop_prob)
        return image, target, index, img_id


    def get_raw_item(self, index):
        if index < self.bp:
            coco, root = self.coco[0], self.root[0]
        else:
            coco, root = self.coco[1], self.root[1]
        ann_id = self.ids[index]
        sentence = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        return root, sentence, img_id, path, image


class CocoDatasetBertTest(data.Dataset):

    def __init__(self, root, json, vocab, split, transform=None, ids=None, drop_prob=0):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        self.train = split == 'train'
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)

        # if ids provided by get_paths, use split-specific ids
        self.ids = list(self.coco.anns.keys()) if ids is None else ids
        self.transform = transform
        self.drop_prob = drop_prob
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        vocab = self.vocab
        root, sentence, img_id, path, image = self.get_raw_item(index)
        if self.transform is not None:
            image = self.transform(image)

        target = process_caption_bert(sentence, self.tokenizer, self.drop_prob, self.train)
        return image, target, index, img_id


    def get_raw_item(self, index):
        if index < self.bp:
            coco, root = self.coco[0], self.root[0]
        else:
            coco, root = self.coco[1], self.root[1]
        ann_id = self.ids[index]
        sentence = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        return root, sentence, img_id, path, image
    
class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None, drop_prob=0):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.drop_prob = drop_prob
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        """
            self.dataset is a list of dictionary with keys like..
                'sentids'
                'imgid'
                'sentences' : list[dict]
                    tokens'
                    raw'
                    imgid'
                    sentid'
                'split'
                'filename'
        """
        self.ids = []
        for i, d in enumerate(self.dataset):
            self.ids += [i] if d['split'] == split else []

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        
        ann_id = self.ids[index]
        img_id = self.ids[index]
        
        captions = [c['raw'] for c in self.dataset[img_id]['sentences']]
        img_path = os.path.join(root, self.dataset[img_id]['filename'])

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        captions = [tokenize(c, vocab, self.drop_prob) for c in captions]
        
        return image, captions, index, img_id

    def __len__(self):
        return len(self.ids)


class FlickrDatasetBert(data.Dataset):
    """
    Dataset loader for Flickr30k datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None, drop_prob=0):
        self.root = root
        self.train = split == 'train'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab
        self.split = split
        self.transform = transform
        self.drop_prob = drop_prob
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        """
            self.dataset is a list of dictionary with keys like..
                'sentids'
                'imgid'
                'sentences' : list[dict]
                    tokens'
                    raw'
                    imgid'
                    sentid'
                'split'
                'filename'
        """
        self.ids = []
        for i, d in enumerate(self.dataset):
            self.ids += [i] if d['split'] == split else []

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        
        ann_id = self.ids[index]
        img_id = self.ids[index]
        
        captions = [c['raw'] for c in self.dataset[img_id]['sentences']]
        img_path = os.path.join(root, self.dataset[img_id]['filename'])

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        captions = [process_caption_bert(c, self.tokenizer, self.drop_prob, self.train) for c in captions]
        
        return image, captions, index, img_id

    def __len__(self):
        return len(self.ids)

class FlickrDatasetTest(data.Dataset):
    """
    Dataset loader for Flickr30k datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None, drop_prob=0):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.drop_prob = drop_prob
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        target = tokenize(caption, vocab, self.drop_prob)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)
    
    def get_raw_item(self, index):
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        sentence = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']
        return root, sentence, None, path, None
    
    
class FlickrDatasetBertTest(data.Dataset):
    """
    Dataset loader for Flickr30k datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None, drop_prob=0):
        self.root = root
        self.train = split == 'train'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab
        self.split = split
        self.transform = transform
        self.drop_prob = drop_prob
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        target = process_caption_bert(caption, self.tokenizer, self.drop_prob, self.train)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)
    
    def get_raw_item(self, index):
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        sentence = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']
        return root, sentence, None, path, None

class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, split, vocab, i_drop_prob, c_drop_prob):
        # FIXME Note that below assertion is essental to prevent using 
        # fast dataset on .npy file with reduduncy (eg. dev.npy, test.npy, testall.npy)
        # This class should be used only with train dataset.
        assert split == 'train'
        self.vocab = vocab
        self.train = split == 'train'
        self.data_path = data_path
        self.data_name = data_name
        self.i_drop_prob = i_drop_prob
        self.c_drop_prob = c_drop_prob

        loc_cap = data_path
        loc_image = data_path

        # Captions
        self.captions = []
        import time
        print('Loading captions from .txt')
        start = time.time()
        with open(os.path.join(loc_cap, '%s_caps.txt' % split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        print('take:', time.time() - start)
        
        print('Loading images from .npy')
        start = time.time()
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % split), mmap_mode='r')
        print('take:', time.time() - start)

        self.length = len(self.images)
        num_images = len(self.images)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        # if num_images != self.length:
        #     self.im_div = 5
        # else:
        #     self.im_div = 1
            
        # the development set for coco is large and so validation would be slow
        if split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        captions = self.captions[index*5:(index+1)*5]

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        targets = [tokenize(c, self.vocab, self.c_drop_prob) for c in captions]
        image = self.images[index]
        
        if self.train:  
            # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > self.i_drop_prob)]
            
        image = torch.tensor(image)
        return image, targets, index, None

    def __len__(self):
        return self.length

    
class PrecompRegionDatasetTest(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, split, vocab, i_drop_prob, c_drop_prob):
        self.vocab = vocab
        self.train = split == 'train'
        self.data_path = data_path
        self.data_name = data_name
        self.i_drop_prob = i_drop_prob
        self.c_drop_prob = c_drop_prob

        loc_cap = data_path
        loc_image = data_path

        # Captions
        self.captions = []
        import time
        print('Loading captions from .txt')
        start = time.time()
        with open(os.path.join(loc_cap, '%s_caps.txt' % split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        print('take:', time.time() - start)
        
        print('Loading images from .npy')
        start = time.time()
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % split), mmap_mode='r')
        print('take:', time.time() - start)

        self.length = len(self.captions)
        num_images = len(self.images)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
            
        # the development set for coco is large and so validation would be slow
        if split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        caption = self.captions[index]

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        targets = tokenize(caption, self.vocab, self.c_drop_prob)
        image = self.images[index//self.im_div]
        
        if self.train:  
            # Size augmentation on region features.
            num_features = image.shape[0]
            idxs_to_drop = random.sample(range(num_features), int(self.i_drop_prob * num_features))            
            image = image[[(not i in idxs_to_drop) for i in range(num_features)]]
            
        image = torch.tensor(image)
        return image, targets, index, None

    def __len__(self):
        return self.length


def process_caption_bert(caption, tokenizer, drop_prob, train):
        output_tokens = []
        deleted_idx = []
        tokens = tokenizer.basic_tokenizer.tokenize(caption)
        
        for i, token in enumerate(tokens):
            sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()

            if prob < drop_prob and train:  # mask/remove the tokens only during training
                prob /= drop_prob

                # 50% randomly change token to mask token
                if prob < 0.5:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.6:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)
                        deleted_idx.append(len(output_tokens) - 1)
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)

        if len(deleted_idx) != 0:
            output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

        output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
        target = tokenizer.convert_tokens_to_ids(output_tokens)
        target = torch.Tensor(target)
        return target


class PrecompRegionDatasetBERT(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, split, vocab, i_drop_prob, c_drop_prob):
        # FIXME Note that below assertion is essental to prevent using 
        # fast dataset on .npy file with reduduncy (eg. dev.npy, test.npy, testall.npy)
        # This class should be used only with train dataset.
        assert split == 'train'
        self.train = split == 'train'
        self.data_path = data_path
        self.data_name = data_name
        self.i_drop_prob = i_drop_prob
        self.c_drop_prob = c_drop_prob

        loc_cap = data_path
        loc_image = data_path

        # Captions
        self.captions = []
        import time
        print('Loading captions from .txt')
        start = time.time()
        with open(os.path.join(loc_cap, '%s_caps.txt' % split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        print('take:', time.time() - start)
        
        print('Loading images from .npy')
        start = time.time()
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % split), mmap_mode='r')
        print('take:', time.time() - start)

        self.length = len(self.images)
        num_images = len(self.images)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab
            
        # the development set for coco is large and so validation would be slow
        if split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        captions = self.captions[index*5:(index+1)*5]

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        targets = [process_caption_bert(c, self.tokenizer, self.c_drop_prob, self.train) for c in captions]
        image = self.images[index]
        
        if self.train:  
            # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > self.i_drop_prob)]
            
        image = torch.tensor(image)
        return image, targets, index, None

    def __len__(self):
        return self.length
    
class PrecompRegionDatasetBERTTest(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, split, vocab, i_drop_prob, c_drop_prob):
        self.train = split == 'train'
        self.data_path = data_path
        self.data_name = data_name
        self.i_drop_prob = i_drop_prob
        self.c_drop_prob = c_drop_prob

        loc_cap = data_path
        loc_image = data_path

        # Captions
        self.captions = []
        import time
        print('Loading captions from .txt')
        start = time.time()
        with open(os.path.join(loc_cap, '%s_caps.txt' % split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        print('take:', time.time() - start)
        
        print('Loading images from .npy')
        start = time.time()
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % split), mmap_mode='r')
        print('take:', time.time() - start)

        self.length = len(self.captions)
        num_images = len(self.images)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab
        
        # the development set for coco is large and so validation would be slow
        if split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        caption = self.captions[index]

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        targets = process_caption_bert(caption, self.tokenizer, self.c_drop_prob, self.train)
        image = self.images[index//self.im_div]
        
        if self.train:  
            # Size augmentation on region features.
            num_features = image.shape[0]
            idxs_to_drop = random.sample(range(num_features), int(self.i_drop_prob * num_features))            
            image = image[[(not i in idxs_to_drop) for i in range(num_features)]]
            
        image = torch.tensor(image)
        return image, targets, index, None

    def __len__(self):
        return self.length


def collate_fn(data):
    """
        input : List of tuples. Each tuple is a output of __getitem__ of the dataset
        output : Collated tensor
    """
    # Sort a data list by sentence length
    images, sentences, img_ids, sentences_ids = zip(*data)
    # compute the number of captions in each images and create match label from it
    flatten_sentences = [sentence for img in list(sentences) for sentence in img]
    flatten_sentences_len = [len(sentence) for sentence in flatten_sentences]
    org_len, org_sen = flatten_sentences_len, flatten_sentences
    caption_data = list(zip(flatten_sentences_len, flatten_sentences))
    sorted_idx = sorted(range(len(caption_data)), key=lambda x: caption_data[x][0], reverse=True)
    recovery_idx = sorted(range(len(caption_data)), key=lambda x: sorted_idx[x], reverse=False)
    caption_data.sort(key=lambda x: x[0], reverse=True)
    flatten_sentences_len, flatten_sentences = zip(*caption_data)
    
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    sentences_len = torch.tensor(flatten_sentences_len)
    recovery_idx = torch.tensor(recovery_idx)
    
    padded_sentences = torch.zeros(len(flatten_sentences), max(sentences_len)).long()
    for i, cap in enumerate(flatten_sentences):
        end = sentences_len[i]
        padded_sentences[i, :end] = cap[:end]

    return images, padded_sentences, sentences_len, recovery_idx, img_ids


def collate_fn_test(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
        data: list of (image, sentence) tuple.
            - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
            - sentence: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256) or 
                        (batch_size, padded_length, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = torch.tensor([len(cap) for cap in sentences])
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, cap_lengths, ids


def collate_fn_butd(data):
    """
        input : List of tuples. Each tuple is a output of __getitem__ of the dataset
        output : Collated tensor
    """
    # Sort a data list by sentence length
    images, sentences, img_ids, sentences_ids = zip(*data)
    # compute the number of captions in each images and create match label from it
    flatten_sentences = [sentence for img in list(sentences) for sentence in img]
    flatten_sentences_len = [len(sentence) for sentence in flatten_sentences]
    org_len, org_sen = flatten_sentences_len, flatten_sentences
    caption_data = list(zip(flatten_sentences_len, flatten_sentences))
    sorted_idx = sorted(range(len(caption_data)), key=lambda x: caption_data[x][0], reverse=True)
    recovery_idx = sorted(range(len(caption_data)), key=lambda x: sorted_idx[x], reverse=False)
    caption_data.sort(key=lambda x: x[0], reverse=True)
    flatten_sentences_len, flatten_sentences = zip(*caption_data)
    
    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    sentences_len = torch.tensor(flatten_sentences_len)
    recovery_idx = torch.tensor(recovery_idx)
    
    padded_sentences = torch.zeros(len(flatten_sentences), max(sentences_len)).long()
    for i, cap in enumerate(flatten_sentences):
        end = sentences_len[i]
        padded_sentences[i, :end] = cap[:end]
    
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images_len = torch.tensor([len(img) for img in images])
    padded_images = torch.zeros(len(images), max(images_len), images[0].shape[-1]).float()
    for i, img in enumerate(images):
        end = images_len[i]
        padded_images[i, :end] = img[:end]

    return padded_images, padded_sentences, images_len, sentences_len, recovery_idx, img_ids


def collate_fn_butd_test(data):
    """
        input : List of tuples. Each tuple is a output of __getitem__ of the dataset
        output : Collated tensor
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, ids, img_ids = zip(*data)
    
    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    sentences_len = torch.tensor([len(cap) for cap in sentences])
    padded_sentences = torch.zeros(len(sentences), max(sentences_len)).long()
    for i, cap in enumerate(sentences):
        end = sentences_len[i]
        padded_sentences[i, :end] = cap[:end]
    
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images_len = torch.tensor([len(img) for img in images])
    padded_images = torch.zeros(len(images), max(images_len), images[0].shape[-1]).float()
    for i, img in enumerate(images):
        end = images_len[i]
        padded_images[i, :end] = img[:end]

    return padded_images, padded_sentences, images_len, sentences_len, ids


def get_loader_single(data_name, split, root, json, vocab, transform,
                                            batch_size=128, shuffle=True, num_workers=2, 
                                            ids=None, opt=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    test = split != 'train'
    caption_drop_prob = opt.caption_drop_prob if split == 'train' else 0
    image_drop_prob = opt.butd_drop_prob if split == 'train' else 0
    coco_butd_splits = {'train':'train', 'val':'dev', 'test':'testall'}
    f30k_butd_splits = {'train':'train', 'val':'dev', 'test':'test'}
    
    use_bert = opt.use_bert
    
    if 'coco' == data_name:
        if test:
            dataset = (CocoDatasetTest if not use_bert else CocoDatasetBertTest)(
                root=root,
                json=json,
                vocab=vocab,
                transform=transform, 
                ids=ids,
                drop_prob=caption_drop_prob,
                split=split)
            collate = collate_fn_test
        else:
            dataset = (CocoDataset if not use_bert else CocoDatasetBert)(
                root=root,
                json=json,
                vocab=vocab,
                transform=transform, 
                ids=ids,
                drop_prob=caption_drop_prob,
                split=split)
            collate = collate_fn
    elif 'f30k' == data_name:
        if test:
            dataset = (FlickrDatasetTest if not use_bert else FlickrDatasetBertTest)(
                root=root,
                json=json,
                split=split,
                vocab=vocab,
                transform=transform,
                drop_prob=caption_drop_prob)
            collate = collate_fn_test
        else:
            dataset = (FlickrDataset if not use_bert else FlickrDatasetBert)(
                root=root,
                json=json,
                split=split,
                vocab=vocab,
                transform=transform,
                drop_prob=caption_drop_prob)
            collate = collate_fn
    elif 'coco_butd' == data_name:
        if test:
            dataset = (PrecompRegionDatasetTest if not use_bert else PrecompRegionDatasetBERTTest)(
                data_path=root,
                data_name='coco_butd',
                split=coco_butd_splits[split],
                vocab=vocab,
                i_drop_prob=image_drop_prob,
                c_drop_prob=caption_drop_prob)
            collate = collate_fn_butd_test
        else:
            dataset = (PrecompRegionDataset if not use_bert else PrecompRegionDatasetBERT)(
                data_path=root,
                data_name='coco_butd',
                split=coco_butd_splits[split],
                vocab=vocab,
                i_drop_prob=image_drop_prob,
                c_drop_prob=caption_drop_prob)
            collate = collate_fn_butd
    elif 'f30k_butd' == data_name:
        if test:
            dataset = (PrecompRegionDatasetTest if not use_bert else PrecompRegionDatasetBERTTest)(
                data_path=root,
                data_name='f30k_butd',
                split=f30k_butd_splits[split],
                vocab=vocab,
                i_drop_prob=image_drop_prob,
                c_drop_prob=caption_drop_prob)
            collate = collate_fn_butd_test
        else:
            dataset = (PrecompRegionDataset if not use_bert else PrecompRegionDatasetBERT)(
                data_path=root,
                data_name='f30k_butd',
                split=f30k_butd_splits[split],
                vocab=vocab,
                i_drop_prob=image_drop_prob,
                c_drop_prob=caption_drop_prob)
            collate = collate_fn_butd
    else:
        assert NotImplementedError

    
    # Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate)
    return data_loader


def get_transform(data_name, split_name, opt):
    return get_image_transform(data_name, split_name, opt)


def get_image_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size) if not opt.gpo_aug else transforms.RandomResizedCrop(512, scale=(0.36, 1))]
        t_list += [transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        if opt.gpo_aug:
            t_list = [transforms.Resize((512, 512))]
        else:
            t_list = [transforms.Resize(int(opt.crop_size/0.875)), transforms.CenterCrop(opt.crop_size)]
    elif split_name == 'test':
        if opt.ten_crop:
            t_list = [transforms.Resize(int(opt.crop_size/0.875)), transforms.TenCrop(opt.crop_size)]
        elif opt.gpo_aug:
            t_list = [transforms.Resize((512, 512))]
        else:
            t_list = [transforms.Resize(int(opt.crop_size/0.875)), transforms.CenterCrop(opt.crop_size)]
    
    if opt.ten_crop:
        t_end = [
            transforms.Lambda(lambda crops: transforms.ToTensor()(crops[opt.ten_crop_idx])),
            normalizer,
            transforms.RandomErasing(p=opt.random_erasing_prob if split_name == 'train' else 0)
        ]
    else:
        t_end = [
            transforms.ToTensor(), 
            normalizer,
            transforms.RandomErasing(p=opt.random_erasing_prob if split_name == 'train' else 0)
        ]
    
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(opt, vocab):
    dpath = os.path.join(opt.data_path, opt.data_name)
    roots, ids = get_paths(dpath, opt.data_name)
    transform = get_transform(opt.data_name, 'train', opt)
    train_loader = get_loader_single(
        opt.data_name, 'train',
        roots['train']['img'], #root
        roots['train']['cap'], #json
        vocab, transform, ids=ids['train'],
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers,
        opt=opt)

    transform = get_transform(opt.data_name, 'val', opt)
    val_loader = get_loader_single(
        opt.data_name, 'val',
        roots['val']['img'],
        roots['val']['cap'],
        vocab, transform, ids=ids['val'],
        batch_size=opt.batch_size_eval, shuffle=False,
        num_workers=opt.workers,
        opt=opt)

    return train_loader, val_loader


def get_test_loader(opt, vocab):
    dpath = os.path.join(opt.data_path, opt.data_name)
    roots, ids = get_paths(dpath, opt.data_name)
    transform = get_transform(opt.data_name, 'test', opt)
    return get_loader_single(
        opt.data_name, 'test',
        roots['test']['img'],
        roots['test']['cap'],
        vocab, transform, ids=ids['test'],
        batch_size=opt.batch_size_eval, shuffle=False,
        num_workers=opt.workers,
        opt=opt)
