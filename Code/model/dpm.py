import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, hidden_size, dim_feedforward, dropout, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.lstm = LSTM(d_model, hidden_size, 1, bidirectional=True)
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_size*2, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear(self.dropout(self.activation(self.lstm(src)[0])))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class DPM_Model(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_spk=2, layer=6, segment_size=250):
        super(DPM_Model, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.eps = 1e-8
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)
        self.DPM = DPM(self.feature_dim, self.hidden_dim, self.feature_dim * self.num_spk, num_layers=layer)

    def pad_segment(self, input, segment_size):
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def split_feature(self, input, segment_size):
        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]
        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]
        return output.contiguous()

    def forward(self, input):
        pass


class BF(DPM_Model):
    def __init__(self, *args, **kwargs):
        super(BF, self).__init__(*args, **kwargs)
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                         nn.Sigmoid()
                                         )
        self.mask_confusion = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, input):
        batch_size, E, seq_length = input.shape
        enc_feature = self.BN(input)
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)
        output = self.DPM(enc_segments).view(batch_size * self.num_spk, self.feature_dim, self.segment_size, -1)
        output = self.merge_feature(output, enc_rest)
        bf_filter = self.output(output) * self.output_gate(output)
        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, self.num_spk, -1, self.feature_dim)
        return bf_filter


class SingleTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(SingleTransformer, self).__init__()
        self.transformer = TransformerEncoderLayer(d_model=input_size, nhead=4, hidden_size=hidden_size,
                                                   dim_feedforward=hidden_size*2, dropout=dropout)

    def forward(self, input):
        output = input
        transformer_output = self.transformer(output.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        return transformer_output


class DPM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(DPM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.row_transformer = nn.ModuleList([])
        self.col_transformer = nn.ModuleList([])
        for i in range(num_layers):
            self.row_transformer.append(SingleTransformer(input_size, hidden_size, dropout))
            self.col_transformer.append(SingleTransformer(input_size, hidden_size, dropout))
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_transformer)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)
            row_output = self.row_transformer[i](row_input)
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            output = row_output
            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)
            col_output = self.col_transformer[i](col_input)
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
            output = col_output
        output = self.output(output)

        return output


class DPM_Net_Model(nn.Module):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, segment_size=250, nspk=2, win_len=2):
        super(DPM_Model, self).__init__()

        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)
        self.separator = BF(self.enc_dim, self.feature_dim, self.hidden_dim,
                                   self.num_spk, self.layer, self.segment_size)
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)

    def pad_input(self, input, window):
        batch_size, nsample = input.shape
        stride = window // 2
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        B, _, _ = input.size()
        score_ = self.enc_LN(input)
        score_ = self.separator(score_)
        score_ = score_.squeeze(1).transpose(1, 2).contiguous()
        score = self.mask_conv1x1(score_)
        est_mask = F.relu(score)
        est_w = input * est_mask
        return est_w