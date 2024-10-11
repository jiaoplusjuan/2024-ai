import gensim
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
import torch
class Sentence:
    def __init__(self, type, vector):
        self.type = int(type)
        self.vector = vector

def load_word2vec_model(filename):
    # 获取模型
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    return model

def build_word_vector_mapping(model):
    # 利用词向量构造字典
    word_vector_mapping = {}
    for word in model.key_to_index:
        word_vector_mapping[word] = model.get_vector(word)
    return word_vector_mapping

input_filename = "Dataset/wiki_word2vec_50.bin"
output_filename = "output.txt"

# 加载Word2Vec模型
model = load_word2vec_model(input_filename)

word_vector_mapping = build_word_vector_mapping(model)
def get_sentence_data(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                type, sentence = parts
                word_vector = sentence.split()  # 将句子拆分成单词
                sentences.append(Sentence(type, word_vector))
    return sentences

def get_word_vector(sentences:List[Sentence]):
    transformed_sentences = []
    for sentence in sentences:
        transformed_vector = []
        for word in sentence.vector:
            if word in word_vector_mapping:
                transformed_vector.append(word_vector_mapping[word])
            else:
                transformed_vector.append(np.zeros(50))
        transformed_sentences.append(Sentence(sentence.type,transformed_vector))
    return transformed_sentences

def get_data_raw(filename):
    sentences = get_sentence_data(filename)
    types = [data.type for data in sentences]
    words = [data.vector for data in sentences]
    texts = []
    for data in sentences:
        text = ''.join(data.vector)
        texts.append(text)
    return texts,types

def get_data2(batch_size):
    texts, labels = get_data_raw("Dataset/train.txt")
    texts2, labels2 = get_data_raw("Dataset/test.txt")
    texts3, labels3 = get_data_raw("Dataset/validation.txt")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    max_length = 200  # 设定最大长度
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                   # 文本
                            add_special_tokens = True, # 添加特殊token，即CLS和SEP
                            max_length = max_length,           # 最大长度
                            padding='max_length',              # 填充
                            truncation=True,                   # 截断
                            return_attention_mask = True,   # 返回attention mask
                            return_tensors = 'pt',          # 返回PyTorch张量
                            
                    )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # 将标签转换为Tensor
    labels = torch.tensor(labels)

    # 将数据封装成TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    input_ids2 = []
    attention_masks2 = []

    for text in texts2:
        encoded_dict = tokenizer.encode_plus(
                            text,                      # 文本
                            add_special_tokens = True, # 添加特殊token，即CLS和SEP
                            max_length = max_length,           # 最大长度
                            padding='max_length',              # 填充
                            truncation=True,                   # 截断
                            return_attention_mask = True,   # 返回attention mask
                            return_tensors = 'pt',          # 返回PyTorch张量
                    )
        
        input_ids2.append(encoded_dict['input_ids'])
        attention_masks2.append(encoded_dict['attention_mask'])

    input_ids2 = torch.cat(input_ids2, dim=0)
    attention_masks2 = torch.cat(attention_masks2, dim=0)

    # 将标签转换为Tensor
    labels2 = torch.tensor(labels2)

    # 将数据封装成TensorDataset
    dataset2 = TensorDataset(input_ids2, attention_masks2, labels2)
    input_ids3 = []
    attention_masks3 = []
    for text in texts3:
        encoded_dict = tokenizer.encode_plus(
                            text,                      # 文本
                            add_special_tokens = True, # 添加特殊token，即CLS和SEP
                            max_length = max_length,           # 最大长度
                            padding='max_length',              # 填充
                            truncation=True,                   # 截断
                            return_attention_mask = True,   # 返回attention mask
                            return_tensors = 'pt',          # 返回PyTorch张量
                    )
        
        input_ids3.append(encoded_dict['input_ids'])
        attention_masks3.append(encoded_dict['attention_mask'])

    input_ids3 = torch.cat(input_ids3, dim=0)
    attention_masks3 = torch.cat(attention_masks3, dim=0)

    # 将标签转换为Tensor
    labels3 = torch.tensor(labels3)

    # 将数据封装成TensorDataset
    dataset3 = TensorDataset(input_ids3, attention_masks3, labels3)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        sampler = RandomSampler(dataset),
        batch_size = batch_size
    )

    dataloader2 = DataLoader(
        dataset2,
        sampler = RandomSampler(dataset2),
        batch_size = batch_size
    )
    
    dataloader3 = DataLoader(
        dataset3,
        sampler = RandomSampler(dataset3),
        batch_size = batch_size
    )
    return dataloader,dataloader2, dataloader3

def showData(filename):
    sentence_data = get_word_vector(get_sentence_data(filename))
    vector_lengths = []  # 用于收集向量长度
    
    for data in sentence_data:
        vector_lengths.append(len(data.vector))  # 收集向量长度
    
    # 绘制向量长度的直方图
    plt.figure(figsize=(10, 5))
    plt.hist(vector_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Vector Lengths')
    plt.xlabel('Vector Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # 保存图表为图片
    plt.savefig('vector_lengths_distribution.png', dpi=300, bbox_inches='tight')
    
    # 关闭图表以释放内存
    plt.close()

showData("./Dataset/train.txt")
