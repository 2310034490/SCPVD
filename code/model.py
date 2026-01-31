# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.autograd as autograd

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x

    
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)
        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [批大小, 序列长度, 嵌入维度]
        embedded = x.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        conved = self.convs(embedded)

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))    # [B, n_filters * len(filter_sizes)]
        return self.fc(cat)

class CNNResNetClassificationSeq(nn.Module):
    """增强型双分支架构：融合CNN和残差网络+多头注意力，采用多层次融合策略"""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.d_size = self.args.d_size
        self.hidden_size = config.hidden_size
        self.num_heads = 8  # 增加注意力头数到8
        self.head_dim = self.hidden_size // self.num_heads

        # 分支1: 增强型CNN
        self.window_size = self.args.cnn_size
        self.filter_size = []
        for i in range(self.args.filter_size):
            i = i+1
            self.filter_size.append(i)
        self.cnn = TextCNN(config.hidden_size, self.window_size, self.filter_size, self.d_size, 0.15)  # 降低dropout
        self.cnn_projection = nn.Linear(self.d_size, self.hidden_size)
        self.cnn_norm = nn.LayerNorm(self.hidden_size)  # CNN特征归一化
        
        # 分支2: 优化的残差网络
        # 深层残差网络层 - 使用3个残差块
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(768 if i == 0 else self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.hidden_size),
                nn.GELU(),  # 使用GELU激活函数
                nn.Dropout(0.2),  # 调整Dropout率
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.hidden_size)
            ) for i in range(3)  # 增加到3个残差块
        ])
        
        # 多头注意力机制相关层
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(0.1)  # 降低注意力dropout
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)  # 层归一化

        self.simple_gate = nn.Linear(self.hidden_size * 2, 1)
        
        # 特征融合层
        self.linear = nn.Linear(self.args.filter_size * config.hidden_size, self.d_size)
        self.dense = nn.Linear(self.hidden_size + self.d_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

        # 类别权重
        self.pos_weight = nn.Parameter(torch.tensor(2.0))  # 正样本权重初始化为2.0
    
    def forward(self, features, **kwargs):
        batch_size = features.shape[0]
        
        # 重塑输入
        x = features.reshape(batch_size, -1, 768)  # [B, L, D]
        
        # 分支1: CNN处理
        cnn_out = self.cnn(x)  # [B, d_size]
        cnn_projected = self.cnn_projection(cnn_out)  # [B, hidden_size]
        cnn_projected = self.cnn_norm(cnn_projected)  # 应用层归一化
        
        # 分支2: 残差网络+多头注意力处理
        # 应用残差网络
        x_resnet = x.transpose(1, 2)  # [B, D, L]
        identity = x_resnet
        
        for i, conv_layer in enumerate(self.conv_layers):
            # 保存原始输入用于残差连接
            identity = x_resnet

            # 应用卷积层
            out = conv_layer(x_resnet)

            # 优化的残差连接 - 如果维度不匹配，进行调整
            if identity.shape[1] != out.shape[1]:
                identity = F.pad(identity, (0, 0, 0, out.shape[1] - identity.shape[1]))

            # 应用残差连接
            out = out + identity
            x_resnet = out
        
        # 将x_resnet从[B, D, L]转换为[B, L, D]以便应用多头注意力
        x_resnet = x_resnet.transpose(1, 2)  # [B, L, D]
        seq_len = x_resnet.shape[1]
        
        # 多头注意力机制实现
        # 1. 线性变换生成查询、键、值
        q = self.query(x_resnet).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        k = self.key(x_resnet).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        v = self.value(x_resnet).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]

        # 2. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]
        attention_weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
        attention_weights = self.attention_dropout(attention_weights)  # 应用dropout防止过拟合

        # 3. 应用注意力权重并合并多头
        context = torch.matmul(attention_weights, v)  # [B, H, L, D/H]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # [B, L, D]

        # 4. 输出投影
        output = self.output_projection(context)  # [B, L, D]

        # 5. 残差连接和层归一化
        output = self.layer_norm(output + x_resnet)  # 残差连接 + 层归一化

        # 聚合特征 - 使用全局平均池化和最大池化的组合
        avg_pooled = torch.mean(output, dim=1)  # [B, D]
        max_pooled, _ = torch.max(output, dim=1)  # [B, D]
        resnet_out = avg_pooled + max_pooled  # 结合两种池化方式 [B, hidden_size]
        
        combined_features = torch.cat([cnn_projected, resnet_out], dim=1)  # [B, 2*hidden_size]
        gate = torch.sigmoid(self.simple_gate(combined_features))
        fused_features = gate * resnet_out + (1.0 - gate) * cnn_projected
        
        # 特征融合
        features_linear = self.linear(features)
        x = torch.cat((fused_features, features_linear), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x


class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3, 1)        # 3->5

        self.cnnclassifier = CNNResNetClassificationSeq(config, self.args)

    def forward(self, seq_ids=None, input_ids=None, labels=None):       
        batch_size = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        token_len = seq_ids.shape[-1]

        seq_inputs = seq_ids.reshape(-1, token_len)                                 # [4, 3, 400] -> [4*3, 400]
        seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]    # [4*3, 400] -> [4*3, 400, 768]
        seq_embeds = seq_embeds[:, 0, :]                                           # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)                           # [4*3, 768] -> [4, 3*768]

        logits_path = self.cnnclassifier(outputs_seq)

        prob_path = torch.sigmoid(logits_path)
        prob = prob_path
        if labels is not None:
            labels = labels.float()
            
            # 改进的损失函数：结合Focal Loss和类别权重
            pos_weight = self.cnnclassifier.pos_weight
            
            # Focal Loss参数
            alpha = 0.25
            gamma = 2.0
            
            # 计算BCE损失
            bce_loss = -pos_weight * labels * torch.log(prob[:, 0] + 1e-10) - (1 - labels) * torch.log((1 - prob)[:, 0] + 1e-10)
            
            # 计算Focal Loss
            pt = torch.where(labels == 1, prob[:, 0], 1 - prob[:, 0])
            focal_weight = alpha * (1 - pt) ** gamma
            focal_loss = focal_weight * bce_loss
            
            # 最终损失
            loss = focal_loss.mean()
            
            return loss, prob
        else:
            return prob
