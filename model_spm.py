import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision

from agg_block.agg_block import AggregationBlock

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange, reduce
from transformers import BertModel
from agg_block.attention import default

def get_cnn(arch, pretrained):
    if arch == 'resnext_wsl':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    else:
        model = torchvision.models.__dict__[arch](pretrained=pretrained) 
    return model

# Problematic: could induce NaN loss
def l2norm_old(x):
    """L2-normalize columns of x"""
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)

def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0)
    if torch.cuda.is_available():
        ind = ind.cuda()
    mask = torch.tensor((ind >= lengths.unsqueeze(1))) if set_pad_to_one \
        else torch.tensor((ind < lengths.unsqueeze(1)))
    return mask.cuda() if torch.cuda.is_available() else mask


def variable_len_pooling(data, input_lens, reduction):
    if input_lens is None:
        if reduction =='avg':
            ret = reduce(data, 'h i k ->  h k', 'mean')
        elif reduction =='max':
            ret = reduce(data, 'h i k ->  h k', 'max')
        elif reduction =='avgmax':
            ret = reduce(data, 'h i k ->  h k', 'max')/2 + reduce(data, 'h i k ->  h k', 'mean')/2
        elif reduction == 'minmax':
            ret = reduce(data, 'h i k ->  h k', 'max')/2 + reduce(data, 'h i k ->  h k', 'min')/2
        else:
            raise NotImplementedError
    else:
        B, N, D = data.shape
        idx = torch.arange(N).unsqueeze(0).expand(B, -1).cuda()
        idx = idx < input_lens.unsqueeze(1)
        idx = idx.unsqueeze(2).expand(-1, -1, D)
        if reduction == 'avg':
            ret = (data * idx.float()).sum(1) / input_lens.unsqueeze(1).float()
        elif reduction == 'max':
            ret = data.masked_fill(~idx, -torch.finfo(data.dtype).max).max(1)[0]
        elif reduction == 'avgmax':
            ret = (data * idx.float()).sum(1) / input_lens.unsqueeze(1).float() +\
                data.masked_fill(~idx, -torch.finfo(data.dtype).max).max(1)[0]
            ret /= 2
        elif reduction == 'minmax':
            ret = data.masked_fill(~idx, torch.finfo(data.dtype).max).min(1)[0] +\
                data.masked_fill(~idx, -torch.finfo(data.dtype).max).max(1)[0]
            ret /= 2
        else:
            raise NotImplementedError
    return ret

class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        if len(x.shape) == 4:
            x = rearrange(x, 'h i j k -> h i (j k)')
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1,2,0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)
        
        output = torch.bmm(attn.transpose(1,2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PIENet(nn.Module):
    """Polysemous Instance Embedding (PIE) module"""

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(PIENet, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, local_feat, global_feat, pad_mask=None, lengths=None):
        residual, attn = self.attention(local_feat, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.num_embeds > 1:
            global_feat = global_feat.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.layer_norm(global_feat + residual)
        return out, attn, residual
        
        
class SetPredictionModule(nn.Module):
    def __init__(
        self, 
        num_embeds, 
        d_in, 
        d_out, 
        axis, 
        pos_enc, 
        query_dim,
        args
    ):
        super(SetPredictionModule, self).__init__()
        self.num_embeds = num_embeds
        self.residual_norm = nn.LayerNorm(d_out) if args.spm_residual_norm else nn.Identity()
        self.spm_residual = args.spm_residual
        self.res_act_local_dropout = nn.Dropout(args.res_act_local_dropout)
        self.res_act_global_dropout = nn.Dropout(args.res_act_global_dropout)
        self.fc = nn.Linear(d_out, 1024) if args.spm_residual_fc else nn.Identity()
        self.res_only_norm = args.res_only_norm
        
        self.agg_block = AggregationBlock(
            depth = args.spm_depth,
            input_channels = d_in,
            input_axis = axis,
            num_latents = num_embeds,
            latent_dim = query_dim,
            num_classes = d_out,
            attn_dropout = args.dropout,
            ff_dropout = args.dropout,
            weight_tie_layers = args.spm_weight_sharing,
            pos_enc_type = pos_enc,
            pre_norm = args.spm_pre_norm,
            post_norm=args.spm_post_norm,
            activation = args.spm_activation,
            last_ln = args.spm_last_ln,
            ff_mult = args.spm_ff_mult,
            more_dropout = args.spm_more_dropout,
            xavier_init = args.spm_xavier_init,
            query_fixed = args.query_fixed,
            query_xavier_init = args.query_xavier_init,
            query_type = 'slot' if args.query_slot else 'learned',
            first_order=args.first_order
        )
        
    def forward(self, local_feat, global_feat=None, pad_mask=None, lengths=None):
        set_prediction = self.agg_block(local_feat, mask=pad_mask)
        set_prediction = self.res_act_local_dropout(set_prediction)
        global_feat = global_feat.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.residual_norm(self.res_act_global_dropout(global_feat)) + set_prediction
        out = self.fc(out)
        
        return out, None, set_prediction


class VSE(nn.Module):
    def __init__(self, word2idx, opt):
        super(VSE, self).__init__()

        self.mil = opt.img_num_embeds >= 1 or opt.txt_num_embeds >= 1
        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(word2idx, opt) if not opt.use_bert else EncoderTextBERT(opt)
        self.amp = opt.amp

    def forward(self, images, sentences, img_len, txt_len):
        with torch.cuda.amp.autocast(enabled=self.amp):
            img_emb, img_attn, img_residual = self.img_enc(images, img_len)
            txt_emb, txt_attn, txt_residual = self.txt_enc(sentences, txt_len)
            return img_emb, txt_emb, img_attn, txt_attn, img_residual, txt_residual
        
        
class SequenceBN(nn.Module):
    def __init__(self, dim, affine=True):
        super(SequenceBN, self).__init__()
        self.bn = nn.BatchNorm1d(dim, affine=affine)
    
    def forward(self, x):
        shape = x.shape
        x = self.bn(rearrange(x, '... d -> (...) d')).reshape(shape)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = SequenceBN(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc2(self.relu(self.bn(self.fc1(x))))
        return x

class EncoderImage(nn.Module):

    def __init__(self, opt, shared_memory=None, shared_query=None):
        super(EncoderImage, self).__init__()
        self.butd = 'butd' in opt.data_name
        embed_size, num_embeds = opt.embed_size, opt.img_num_embeds
        self.grid_drop_prob = opt.grid_drop_prob
        self.global_feat_holder = nn.Identity()

        if not self.butd:
            # Backbone CNN
            self.cnn = get_cnn(opt.cnn_type, True)
            local_feat_dim = self.local_feat_dim = self.cnn.fc.in_features
            self.cnn.avgpool = nn.Sequential()
            self.cnn.fc = nn.Sequential()
        else:
            self.cnn = nn.Identity()
            local_feat_dim = self.local_feat_dim = 2048

        self.img_1x1_dropout = nn.Dropout(opt.img_1x1_dropout)
        self.fc = nn.Linear(local_feat_dim, embed_size)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc = nn.Linear(local_feat_dim, opt.embed_size)
        if opt.gpo_1x1:
            self.mlp = MLP(opt.embed_size, opt.embed_size // 2, opt.embed_size)
            self.spm_fc = lambda x: self.img_1x1_dropout(self.mlp(self.fc(x)) + self.fc(x))
        else:
            self.mlp = None
            self.spm_fc = self.fc if opt.spm_1x1 else nn.Identity()
        
        if 'slot' == opt.arch:
            self.spm = SetPredictionModule(
                num_embeds=num_embeds, 
                d_in=opt.spm_input_dim if opt.spm_1x1 else 2048, 
                d_out=embed_size, 
                axis=2, 
                pos_enc=opt.spm_img_pos_enc_type, 
                query_dim=opt.spm_query_dim,
                args=opt
            )
        elif 'pvse' == opt.arch:
            self.spm = PIENet(num_embeds, opt.spm_input_dim, embed_size, opt.spm_input_dim // 2)
            
        self.residual = opt.spm_residual
        assert opt.img_res_first_fc or opt.img_res_last_fc
        assert opt.img_res_pool in ['avg', 'max']
        
        self.img_res_pool = opt.img_res_pool
        self.inter_dim = opt.spm_input_dim if opt.img_res_first_fc else local_feat_dim
        
        self.residual_first_fc = self.spm_fc if opt.img_res_first_fc else nn.Identity()
        self.residual_last_fc = nn.Linear(self.inter_dim, embed_size) if opt.img_res_last_fc else nn.Identity()
        self.residual_first_pool = variable_len_pooling if opt.img_res_first_pool else lambda x, y, z: x
        self.residual_after_pool = variable_len_pooling if not opt.img_res_first_pool else lambda x, y, z: x
        if opt.img_res_last_pool:
            self.residual_first_pool = lambda x, y, z: x
            self.residual_after_pool = lambda x, y, z: x
        self.residual_last_pool = variable_len_pooling if opt.img_res_last_pool else lambda x, y, z: x
        
        assert opt.spm_img_pos_enc_type == 'none' if self.butd else True

        if opt.spm_xavier_init:
            self.init_weights()
        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = opt.img_finetune

    def residual_connection(self, x, l):
        x = rearrange(x, 'h i j k -> h (i j) k')
        x = self.residual_first_fc(self.residual_first_pool(x, l, self.img_res_pool))
        x = self.residual_last_fc(self.residual_after_pool(x, l, self.img_res_pool))
        x = self.residual_last_pool(x, l, self.img_res_pool)
        return x

    def init_weights(self):
        def fn(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.mlp.apply(fn)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def grid_feature_drop(self, x):
        # This function requires X with shape of B (...) D; N-dimensional tensor but could be squeezed to 3D
        x = rearrange(x, 'b h w d -> b (h w) d')
        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = x.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(x.size(0), x.size(1))
            rand_list_2 = np.random.rand(x.size(0))
            for i in range(x.size(0)):
                if rand_list_2[i] > self.grid_drop_prob:
                    feat_i = x[i][np.where(rand_list_1[i] > self.grid_drop_prob * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, x.size(-1)).to(x.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = x[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            x = torch.stack(features, dim=0)
            x = x[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(x.device)
        else:
            feat_lengths = torch.zeros(x.size(0)).to(x.device)
            feat_lengths[:] = x.size(1)

        return rearrange(x, 'b n d -> b 1 n d'), feat_lengths

    def forward(self, images, lengths=None):
        if not self.butd:
            out_nxn = self.cnn(images)
            s = out_nxn.shape[-1]
            out_nxn = out_nxn.view(
                -1, 
                self.local_feat_dim, 
                int((s/self.local_feat_dim)**0.5), 
                int((s/self.local_feat_dim)**0.5)
            )
            pad_mask = None
        else:
            out_nxn = rearrange(images, 'b (n 1) d -> b d n 1')
            pad_mask = get_pad_mask(images.shape[1], lengths, True)
            
        out_nxn = rearrange(out_nxn, 'h i j k -> h j k i')
        if self.grid_drop_prob > 0:
            out_nxn, lengths = self.grid_feature_drop(out_nxn)
        out, attn, residual = self.spm(
            local_feat=self.spm_fc(out_nxn), 
            global_feat=self.global_feat_holder(self.residual_connection(out_nxn, lengths)),
            pad_mask=pad_mask,
            lengths=lengths
        )
        
        out = l2norm(out)

        return out, attn, residual



class EncoderText(nn.Module):

    def __init__(self, word2idx, opt, shared_memory=None, shared_query=None):
        super(EncoderText, self).__init__()

        wemb_type, word_dim, embed_size, num_embeds = \
            opt.wemb_type, opt.word_dim, opt.embed_size, opt.txt_num_embeds

        self.embed_size = embed_size
        self.use_attention = opt.txt_attention

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)
        self.embed.weight.requires_grad = opt.txt_finetune

        # Sentence embedding
        self.gpo_rnn = opt.gpo_rnn
        self.rnn_hidden_size = embed_size if opt.gpo_rnn else embed_size // 2
        self.rnn = nn.GRU(word_dim, self.rnn_hidden_size, bidirectional=True, batch_first=True)
        
        self.txt_attention_input = opt.txt_attention_input
        self.txt_pooling = opt.txt_pooling
        self.txt_pooling_fc = nn.Linear(embed_size, word_dim) if opt.txt_pooling_fc else nn.Identity()
        assert self.txt_attention_input in ['wemb', 'rnn']
        assert self.txt_pooling in ['rnn', 'max']
        
        self.txt_attention_head = opt.arch
        self.txt_attention_input_dim = word_dim if self.txt_attention_input == 'wemb' \
            else embed_size
        self.residual = opt.spm_residual
        
        if opt.arch == 'pvse':
            self.spm = PIENet(num_embeds, self.txt_attention_input_dim, embed_size, word_dim//2, opt.dropout)
        elif opt.arch == 'slot':
            self.spm = SetPredictionModule(
                num_embeds=num_embeds, 
                d_in=self.txt_attention_input_dim, 
                d_out=embed_size, 
                axis=1, 
                pos_enc=opt.spm_txt_pos_enc_type, 
                query_dim=opt.spm_query_dim,
                args=opt
            )
        else:
            raise NotImplementedError("Invalid attention head for text modality.")
        
        self.dropout = nn.Dropout(opt.dropout if not opt.rnn_no_dropout else 0)

        self.init_weights(wemb_type, word2idx, word_dim)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if 'none' == wemb_type:
            # nn.init.xavier_uniform_(self.embed.weight)
            self.embed.weight.data.uniform_(-0.1, 0.1)
            print('No wemb init. with pre-trained weight')
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                import torchtext
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                import torchtext
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-','').replace('.','').replace("'",'')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx)-len(missing_words), len(word2idx), len(missing_words)))
            
    def residual_connection(self, rnn_out, rnn_out_last, lengths):
        if self.txt_pooling == 'rnn':
            ret = rnn_out_last
        elif self.txt_pooling == 'max':
            ret = variable_len_pooling(rnn_out, lengths, reduction='max')
        ret = self.txt_pooling_fc(ret)
        return ret

    def forward(self, x, lengths):
        # Embed word ids to vectors
        wemb_out = self.embed(x)
        wemb_out = self.dropout(wemb_out)

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths.cpu(), batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        
        rnn_out, rnn_out_last = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        rnn_out_last = rnn_out_last.permute(1, 0, 2).contiguous()
        rnn_out = pad_packed_sequence(rnn_out, batch_first=True)[0]
        if self.gpo_rnn:
            rnn_out_last = (rnn_out_last[:, 0, :] + rnn_out_last[:, 1, :]) / 2
            rnn_out = (rnn_out[:, :, :rnn_out.shape[-1] // 2] + rnn_out[:, :, rnn_out.shape[-1] // 2:]) / 2
        else:
            rnn_out_last = rnn_out_last.view(-1, self.embed_size)

        rnn_out, rnn_out_last = self.dropout(rnn_out), self.dropout(rnn_out_last)

        pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
        if self.txt_attention_head == 'pvse':
            out, attn, residual = self.spm(local_feat=wemb_out, global_feat=rnn_out_last, pad_mask=pad_mask)
        elif self.txt_attention_head == 'slot':
            out, attn, residual = self.spm(
                local_feat=rnn_out if self.txt_attention_input == 'rnn' else wemb_out,
                global_feat=self.residual_connection(rnn_out, rnn_out_last, lengths) if self.residual else None, 
                pad_mask=pad_mask,
                lengths=lengths
            )
        
        out = l2norm(out)
        return out, attn, residual

    
class EncoderTextBERT(nn.Module):

    def __init__(self, opt, shared_memory=None, shared_query=None):
        super(EncoderTextBERT, self).__init__()

        wemb_type, word_dim, embed_size, num_embeds = \
            opt.wemb_type, opt.word_dim, opt.embed_size, opt.txt_num_embeds

        self.embed_size = embed_size
        self.use_attention = opt.txt_attention
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, self.embed_size)
        self.use_checkpoint = opt.use_checkpoint

        # Sentence embedding
        self.txt_attention_input = opt.txt_attention_input
        self.txt_pooling = opt.txt_pooling
        assert self.txt_pooling in ['cls', 'max']
        
        self.txt_attention_head = opt.arch
        self.txt_attention_input_dim = embed_size
        self.residual = opt.spm_residual
        self.sep_bert_fc = opt.sep_bert_fc
        if self.sep_bert_fc:
            self.linear2 = nn.Linear(768, self.embed_size)
        
        if opt.arch == 'pvse':
            self.spm = PIENet(
                num_embeds, 
                self.txt_attention_input_dim, 
                embed_size, 
                word_dim//2, 
                opt.dropout
            )
        elif opt.arch == 'slot':
            self.spm = SetPredictionModule(
                num_embeds=num_embeds, 
                d_in=self.txt_attention_input_dim, 
                d_out=embed_size, 
                axis=1, 
                pos_enc=opt.spm_txt_pos_enc_type, 
                query_dim=opt.spm_query_dim,
                args=opt
            )
        else:
            raise NotImplementedError("Invalid attention head for text modality.")
        
        self.dropout = nn.Dropout(opt.dropout if not opt.rnn_no_dropout else 0)

    
    def residual_connection(self, bert_out, bert_out_cls, lengths):
        if self.txt_pooling == 'cls':
            ret = bert_out_cls
        elif self.txt_pooling == 'max':
            ret = variable_len_pooling(bert_out, lengths, reduction='max')
        return ret

    def forward(self, x, lengths):
        bert_attention_mask = (x != 0).float()
        pie_attention_mask = (x == 0)
        bert_emb = self.bert(x, bert_attention_mask)
        bert_emb = bert_emb[0]
        cap_len = lengths
        
        local_cap_emb = self.linear(bert_emb)
        global_cap_emb = self.residual_connection(
            local_cap_emb if not self.sep_bert_fc else self.linear2(bert_emb), 
            local_cap_emb[:, 0] if not self.sep_bert_fc else self.linear2(bert_emb)[:, 0], 
            cap_len
        )

        if self.txt_attention_head == 'pvse':
            out, attn, residual = self.spm(local_feat=local_cap_emb, global_feat=global_cap_emb, pad_mask=pie_attention_mask)
        elif self.txt_attention_head == 'slot':
            out, attn, residual = self.spm(
                local_feat=local_cap_emb,
                global_feat=global_cap_emb, 
                pad_mask=pie_attention_mask,
                lengths=lengths
            )
        
        out = l2norm(out)
        
        return out, attn, residual
