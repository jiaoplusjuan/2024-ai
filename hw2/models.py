import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, dropout_prob=0.3):
        super(TextCNN, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 50, kernel_size=(k, 50))  # 每个卷积层都有100个独立的卷积核
            for k in kernel_sizes
        ])
        
        for conv_layer in self.conv_layers:
            nn.init.kaiming_normal_(conv_layer.weight)
        
        self.dropout = nn.Dropout(dropout_prob)  # dropout层
        
        self.fc = nn.Linear(len(kernel_sizes) * 50, out_channels)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        pooled_outputs = []  # 使用全零张量初始化列表

        # print(x.shape)
        for conv_layer in self.conv_layers:
            # x:batch_size * 1 * word_length * vector_length
            conv_out = conv_layer(x)
            # conv_out:batch_size * num_filters * changed_word_length * 1
            
            pooled_out, _ = torch.max(conv_out, dim=2)
            pooled_out = pooled_out.view(pooled_out.size(0), -1, 1)  
            # pooled_out:batch_size * num_filters * 1
            pooled_outputs.append(pooled_out)
        pooled_outputs = torch.cat(pooled_outputs, dim=1)
        pooled_outputs = pooled_outputs.squeeze(dim=2)
        # 应用dropout
        pooled_outputs = self.dropout(pooled_outputs)
        # 应用全连接层
        output = self.fc(pooled_outputs)
        return output

def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1) 

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层全连接层
        self.dropout = nn.Dropout(dropout_p)  # Dropout层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 第三层全连接层
        # 初始化权重参数
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        batch_size = x.size(0)  # 获取第一个维度的大小，即批量大小
        x = x.view(batch_size, -1)
        # 改变形状
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数的第一层
        x = self.dropout(x)  # 应用dropout
        x = self.fc2(x)  # 第三层输出层
        return x
    
# 定义 LSTM 模型
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers=2):
        super(RNN_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.squeeze()
        # 去掉第二维
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out

class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size,  output_size,num_layers=2):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.squeeze()
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 GRU
        out, _ = self.gru(x, h0)
        
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

class CustomBERT(nn.Module):
    def __init__(self, num_classes, hidden_size=768):
        super(CustomBERT, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # 随机初始化权重
        self.bert = nn.Embedding(30522, hidden_size)  # 这里使用随机初始化的Embedding层来代替BERT模型
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_vectors):
        input_vectors = torch.squeeze(input_vectors, dim=1)

        # 将input_vectors转换为Bert模型的输入
        embedded = self.bert(input_vectors)
        
        # 取平均池化特征
        pooled_output = torch.mean(embedded, dim=1)

        # 应用dropout
        pooled_output = self.dropout(pooled_output)

        # 使用全连接层进行分类
        logits = self.fc(pooled_output)

        return logits
    
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义双向GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2是因为双向RNN有两个方向

    def forward(self, input):
        input=input.float()
        # print(input.shape)
        input = input.view(input.size(0), -1, input.size(-1))  # 将最后两维压缩成一个维度
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(input.device)  # *2是因为双向RNN有两个方向

        # 正向RNN
        out, _ = self.gru(input, h0)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        return out


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.num_heads = 8  # 设置注意力头数
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, self.num_heads, hidden_size, 0.2),
            num_layers=4
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.squeeze(1)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x

