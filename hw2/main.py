from dataLoader import *
from typing import List
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import *
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from lion_pytorch import Lion
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from photo import *
from sklearn.metrics import f1_score
import torch.optim.lr_scheduler as lr_scheduler
test_data_loader = None
val_data_loader = None
train_data_loader = None
print_every = 100
dtype = torch.float32 
device = None
word_vector_mapping={}
model_path = "Dataset/wiki_word2vec_50.bin"
max_length = 50

class CustomDataset(Dataset):
    def __init__(self, word_vectors, labels):
        self.word_vectors = word_vectors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        word_vector = self.word_vectors[idx]
        label = self.labels[idx]

        return word_vector, label

class RawDataset(Dataset):
    def __init__(self, word_vectors, labels, tokenizer):
        self.word_vectors = word_vectors
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sentence = self.tokenizer.decode(self.word_vectors[idx])
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label

def init_dict():
    # 导入分词到词向量的字典
    global word_vector_mapping
    model = load_word2vec_model(model_path)
    word_vector_mapping = build_word_vector_mapping(model)

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
    
def getData(filename):
    sentence_data = get_word_vector(get_sentence_data(filename))
    vectors = []
    for data in sentence_data:
        vector = np.array(data.vector, dtype=np.float32)
        if len(vector) < max_length:
            # 创建零向量填充
            zero_padding = np.zeros((max_length - len(vector), len(vector[0])), dtype=np.float32)
            padded_vector = np.concatenate((vector, zero_padding), axis=0)
        else:
            # 截取前 max_length 个元素
            padded_vector = vector[:max_length]
        vectors.append(padded_vector)
    types = [data.type for data in sentence_data]
    vectors_np = np.array(vectors)
    types_tensor = torch.tensor(types)
    vectors_tensor = torch.tensor(vectors_np)
    vectors_tensor = vectors_tensor.unsqueeze(1)
    print(vectors_tensor.shape)
    dataset = CustomDataset(vectors_tensor, types_tensor)
    return dataset

def evaluate_model_normal(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # 将模型置于评估模式
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return acc

def get_labels_and_predictions_normal(data_loader, model):
    model.eval()  # 将模型置于评估模式
    labels = []
    predictions = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)  # 获取预测的类别索引
            labels.extend(target.cpu().numpy())  # 将真实标签添加到列表中
            predictions.extend(predicted.cpu().numpy())  # 将预测标签添加到列表中
            
    return labels, predictions

def get_labels_and_predictions_BERT(data_loader, model):
    model.eval()  # 将模型置于评估模式
    labels = []
    predictions = []
    with torch.no_grad():
        for data, attention_mask, target in data_loader:  # 加载注意力掩码数据
            data = data.to(device)
            attention_mask = attention_mask.to(device)  # 将注意力掩码移到相同的设备上
            output = model(data, attention_mask=attention_mask)  # 使用注意力掩码进行模型预测
            output_logits = output.logits  # 假设logits是预测结果
            _, predicted = torch.max(output_logits, 1)
            labels.extend(target.cpu().numpy())  # 将真实标签添加到列表中
            predictions.extend(predicted.cpu().numpy())  # 将预测标签添加到列表中
            
    return labels, predictions

def evaluate_model_loss(model, data_loader):
    model.eval()  # 将模型置于评估模式
    epoch_loss = 0.0
    with torch.no_grad():  # 在评估期间不计算梯度
        for data, target in data_loader:
            data, target = data.to(device=device), target.to(device=device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)  # 计算平均损失

def train(model, optimizer, epochs=20, name=None, scheduler=None):
    model = model.to(device=device)
    
    train_losses = []
    val_losses = []  # 存储验证集损失
    test_accuracies = []
    val_accuracies = []
    train_accuracies = []
    f1_scores = []
    
    for epoch in range(epochs):
        model.train()  # 将模型置于训练模式
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(device=device), target.to(device=device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_data_loader)
        train_losses.append(epoch_loss)
        
        # 计算验证集上的损失和准确率
        val_loss = evaluate_model_loss(model, val_data_loader)
        val_losses.append(val_loss)
        
        # 计算测试集上的准确率
        test_accuracy = evaluate_model_normal(test_data_loader, model)
        val_accuracy = evaluate_model_normal(val_data_loader, model)
        train_accuracy = evaluate_model_normal(train_data_loader, model)
        
        # 计算并存储 F1 分数
        test_labels, test_predictions = get_labels_and_predictions_normal(test_data_loader, model)
        test_f1 = f1_score(test_labels, test_predictions)
        f1_scores.append(test_f1)
        
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        train_accuracies.append(train_accuracy)
        
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Epoch {epoch + 1}/{epochs}, Test Accuracy: {test_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        if scheduler is not None:
            scheduler.step()
    min_val_loss_index = val_losses.index(min(val_losses))
    final_test_accuracy = test_accuracies[min_val_loss_index]

    return train_losses, val_losses, train_accuracies, test_accuracies, val_accuracies, f1_scores, final_test_accuracy

def evaluate_model_BERT(model, testdataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in testdataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def evaluate_model_BERT_losses(model, data_loader, device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    with torch.no_grad():  # 在评估期间不计算梯度
        for i, batch in enumerate(data_loader):
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            
            # 累加损失
            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def train2(epochs, model, optimizer, name):
    test_accuracies = []  # 存储测试正确率
    val_accuracies = []  # 存储验证集正确率
    train_accuracies = []  # 存储训练集正确率
    losses = []  # 用于存储每个epoch的训练损失
    val_losses = []  # 存储验证集的损失
    f1_scores = []
    for epoch in range(epochs):
        print("Epoch:", epoch+1)
        model = model.to(device)
        model.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(train_data_loader)):
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_data_loader)
        losses.append(avg_train_loss)  # 存储每个epoch的训练损失
        print("Average training loss:", avg_train_loss)
        val_loss = evaluate_model_BERT_losses(model, val_data_loader, device)
        val_losses.append(val_loss)
        print("Validation Loss:", val_loss)
        # 评估模型在测试集上的准确率
        test_accuracy = evaluate_model_BERT(model, test_data_loader, device)
        train_accuracy = evaluate_model_BERT(model, train_data_loader, device)
        val_accuracy = evaluate_model_BERT(model, val_data_loader, device)
        test_labels, test_predictions = get_labels_and_predictions_BERT(test_data_loader, model)
        test_f1 = f1_score(test_labels, test_predictions)
        f1_scores.append(test_f1)
        val_accuracies.append(val_accuracy)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print("Test Accuracy:", test_accuracy)
    
    min_val_loss_index = val_losses.index(min(val_losses))
    final_test_accuracy = test_accuracies[min_val_loss_index]
    return losses, val_losses,train_accuracies, test_accuracies, val_accuracies, f1_scores, final_test_accuracy

def init_data(batch_size, origin=False):
    global train_data_loader, test_data_loader, val_data_loader
    if origin:
        train_data_loader, test_data_loader, val_data_loader = get_data2(batch_size)
        print("Successfully loading raw data")
    else:
        test_dataset = getData("Dataset/test.txt")
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        train_dataset = getData("Dataset/train.txt")
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = getData("Dataset/validation.txt")
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        print("Successfully loading data")

def main():
    global dtype, device
    parser = argparse.ArgumentParser(description='Train a model.')
    
    # 参数定义
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model', type=str, default='TextCNN', help='Model to use')
    args = parser.parse_args()

    USE_GPU = args.gpu
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    model_name = args.model
    dtype = torch.float32 # We will be using float throughout this tutorial.
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using PyTorch version:", torch.__version__, " Device:", device)
    init_dict()
    
    model = None
    scheduler = None
    single_train_losses = []
    single_val_losses = []
    single_train_accuracies = []
    single_val_accuracies = []
    single_test_accuracies = []
    single_f1_scores = []
    if model_name == 'TextCNN':
        model = TextCNN(in_channels=1, out_channels=2, kernel_sizes=[3, 4, 5])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        init_data(batch_size)
    elif model_name == 'RNN_LSTM':
        model = RNN_LSTM(input_size=50, hidden_size=100,output_size = 2, num_layers=2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        init_data(batch_size)
    elif model_name == 'RNN_GRU':
        model = RNN_GRU(input_size=50, hidden_size=100, output_size = 2,num_layers=2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        init_data(batch_size)
    elif model_name == 'MLP':
        model = MLP(input_size=50*max_length, hidden_size=2000,output_size=2)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
        init_data(batch_size)
    elif model_name == 'BIRNN':
        model = BiRNN(input_size=50, hidden_size=100,output_size=2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
        init_data(batch_size)
    elif model_name == 'BERT':
        model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
        init_data(batch_size, origin=True)
        single_train_losses, single_val_losses, single_train_accuracies, single_test_accuracies, single_val_accuracies, single_f1_scores, accuracy=train2(epochs, model, optimizer, model_name)
        plot_model(single_train_losses, single_val_losses, single_train_accuracies, single_test_accuracies, single_val_accuracies, single_f1_scores, model_name)
        return
    elif model_name == 'Transformer':
        model = Transformer(input_size=50, hidden_size=200,output_size=2)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
        init_data(batch_size)
    elif model_name == 'ALL': # 完成所有模型的绘制任务
        model_TEXTCNN = TextCNN(in_channels=1, out_channels=2, kernel_sizes=[3, 4, 5])
        model_RNN_LSTM = RNN_LSTM(input_size=50, hidden_size=100, output_size = 2)
        model_RNN_GRU = RNN_GRU(input_size=50, hidden_size=100, num_layers=2,output_size = 2)
        model_MLP = MLP(input_size=50*max_length, hidden_size=2000,output_size=2)
        model_BIRNN = BiRNN(input_size=50, hidden_size=100,output_size=2)
        model_Tranformer = Transformer(input_size=50, hidden_size=200,output_size=2)
        optimizer_TEXTCNN = optim.Adam(model_TEXTCNN.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer_RNN_LSTM = optim.Adam(model_RNN_LSTM.parameters(), lr=1e-3)
        optimizer_RNN_GRU = optim.Adam(model_RNN_GRU.parameters(), lr=2e-3)
        optimizer_MLP = optim.Adam(model_MLP.parameters(), lr=learning_rate, weight_decay=1e-3)
        optimizer_BIRNN = optim.Adam(model_BIRNN.parameters(), lr=2e-3)
        optimizer_Transformer = optim.AdamW(model_Tranformer.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler_MLP = lr_scheduler.StepLR(optimizer_MLP, step_size=10, gamma=0.3)
        scheduler_Transformer = lr_scheduler.StepLR(optimizer_Transformer, step_size=10)
        init_data(batch_size)
        Transformer_train_losses, Transformer_val_losses, Transformer_train_accuracies, Transformer_test_accuracies, Transformer_val_accuracies ,Transformer_f1_scores, Transformer_accuracy= train(model_Tranformer, optimizer_Transformer,epochs, model_name,scheduler_Transformer)
        model_BERT = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
        TEXTCNN_train_losses,  TEXTCNN_val_losses, TEXTCNN_train_accuracies, TEXTCNN_test_accuracies, TEXTCNN_val_accuracies, TextCNN_f1_scores, TextCNN_accuracy = train(model_TEXTCNN, optimizer_TEXTCNN,epochs, model_name)
        RNN_LSTM_train_losses, RNN_LSTM_val_losses, RNN_LSTM_train_accuracies, RNN_LSTM_test_accuracies, RNN_LSTM_val_accuracies, RNN_LSTM_f1_scores, RNN_LSTM_accuracy = train( model_RNN_LSTM, optimizer_RNN_LSTM,epochs, model_name)
        RNN_GRU_train_losses, RNN_GRU_val_losses, RNN_GRU_train_accuracies, RNN_GRU_test_accuracies, RNN_GRU_val_accuracies, RNN_GRU_f1_scores, RNN_GRU_accuracy = train(model_RNN_GRU, optimizer_RNN_GRU,epochs, model_name)
        MLP_train_losses, MLP_val_losses, MLP_train_accuracies, MLP_test_accuracies, MLP_val_accuracies ,MLP_f1_scores, MLP_accuracy= train(model_MLP, optimizer_MLP,epochs, model_name, scheduler_MLP)
        BIRNN_train_losses, BIRNN_val_losses, BIRNN_train_accuracies, BIRNN_test_accuracies, BIRNN_val_accuracies , BIRNN_f1_scores, BIRNN_accuracy= train(model_BIRNN, optimizer_BIRNN,epochs, model_name)
        
        optimizer_BERT = AdamW(model_BERT.parameters(), lr=1e-5, weight_decay=0.1)
            
        if batch_size >= 50:
            batch_size = 50
        init_data(batch_size, origin=True)
        BERT_train_losses, BERT_val_losses,BERT_train_accuracies, BERT_test_accuracies, BERT_val_accuracies, BERT_f1_scores, BERT_accuracy = train2(epochs,model_BERT, optimizer_BERT, model_name)
        losses_dict = {
            'TEXTCNN': TEXTCNN_train_losses,
            'RNN_LSTM': RNN_LSTM_train_losses,
            'RNN_GRU': RNN_GRU_train_losses,
            'MLP': MLP_train_losses,
            'BIRNN': BIRNN_train_losses,
            'BERT': BERT_train_losses,
            'Transformer': Transformer_train_losses,
        }
            
        losses_val_dict = {
            'TEXTCNN': TEXTCNN_val_losses,
            'RNN_LSTM': RNN_LSTM_val_losses,
            'RNN_GRU': RNN_GRU_val_losses,
            'MLP': MLP_val_losses,
            'BIRNN': BIRNN_val_losses,
            'BERT': BERT_val_losses,
            'Transformer': Transformer_val_losses,
        }

        test_accuracies = {
            'TEXTCNN': TEXTCNN_test_accuracies,
            'RNN_LSTM': RNN_LSTM_test_accuracies,
            'RNN_GRU': RNN_GRU_test_accuracies,
            'MLP': MLP_test_accuracies,
            'BIRNN': BIRNN_test_accuracies,
            'BERT': BERT_test_accuracies,
            'Transformer': Transformer_test_accuracies,
        }

        val_accuracies = {
            'TEXTCNN': TEXTCNN_val_accuracies,
            'RNN_LSTM': RNN_LSTM_val_accuracies,
            'RNN_GRU': RNN_GRU_val_accuracies,
            'MLP': MLP_val_accuracies,
            'BIRNN': BIRNN_val_accuracies,
            'BERT': BERT_val_accuracies,
            'Transformer': Transformer_val_accuracies,
        }
            
        train_accuracies = {
            'TEXTCNN': TEXTCNN_train_accuracies,
            'RNN_LSTM': RNN_LSTM_train_accuracies,
            'RNN_GRU': RNN_GRU_train_accuracies,
            'MLP': MLP_train_accuracies,
            'BIRNN': BIRNN_train_accuracies,
            'BERT': BERT_train_accuracies,
            'Transformer': Transformer_train_accuracies,
        }
            
        f1_scores = {
            'TEXTCNN':TextCNN_f1_scores,
            'RNN_LSTM': RNN_LSTM_f1_scores,
            'RNN_GRU': RNN_GRU_f1_scores,
            'MLP': MLP_f1_scores,
            'BIRNN': BIRNN_f1_scores,
            'BERT': BERT_f1_scores,
            'Transformer': Transformer_f1_scores,
        }
            
        plot_losses_train(losses_dict)
        plot_losses_val(losses_val_dict)
        plot_test_accuracies(test_accuracies)
        plot_val_accuracies(val_accuracies)
        plot_train_accuracies(train_accuracies)
        plot_f1_scores(f1_scores)
        plot_model(TEXTCNN_train_losses, TEXTCNN_val_losses, TEXTCNN_train_accuracies, TEXTCNN_test_accuracies, TEXTCNN_val_accuracies, TextCNN_f1_scores,'TEXTCNN')
        plot_model(RNN_LSTM_train_losses, RNN_LSTM_val_losses, RNN_LSTM_train_accuracies, RNN_LSTM_test_accuracies, RNN_LSTM_val_accuracies, RNN_LSTM_f1_scores,'RNN_LSTM')
        plot_model(RNN_GRU_train_losses, RNN_GRU_val_losses, RNN_GRU_train_accuracies, RNN_GRU_test_accuracies, RNN_GRU_val_accuracies, RNN_GRU_f1_scores,'RNN_GRU')
        plot_model(MLP_train_losses, MLP_val_losses, MLP_train_accuracies, MLP_test_accuracies, MLP_val_accuracies, MLP_f1_scores,'MLP')
        plot_model(BIRNN_train_losses, BIRNN_val_losses, BIRNN_train_accuracies, BIRNN_test_accuracies, BIRNN_val_accuracies, BIRNN_f1_scores,'BIRNN')
        plot_model(BERT_train_losses, BERT_val_losses, BERT_train_accuracies, BERT_test_accuracies, BERT_val_accuracies, BERT_f1_scores,'BERT')
        plot_model(Transformer_train_losses, Transformer_val_losses, Transformer_train_accuracies, Transformer_test_accuracies, Transformer_val_accuracies, Transformer_f1_scores,'Transformer')
        print('TEXTCNN final accuracy:', TextCNN_accuracy)
        print('RNN_LSTM final accuracy:', RNN_LSTM_accuracy)
        print('RNN_GRU final accuracy:', RNN_GRU_accuracy)
        print('MLP final accuracy:', MLP_accuracy)
        print('BIRNN final accuracy:', BIRNN_accuracy)
        print('BERT final accuracy:', BERT_accuracy)
        print('Transformer final accuracy:', Transformer_accuracy)
        return
    else:
        raise ValueError("Invalid model name.")
    if scheduler:
        single_train_losses, single_val_losses, single_train_accuracies, single_test_accuracies, single_val_accuracies, single_f1_scores, accuracy = train(model, optimizer, epochs,model_name, scheduler)
    else:
        single_train_losses, single_val_losses, single_train_accuracies, single_test_accuracies, single_val_accuracies, single_f1_scores, accuracy = train(model, optimizer, epochs,model_name)
    plot_model(single_train_losses, single_val_losses, single_train_accuracies, single_test_accuracies, single_val_accuracies, single_f1_scores,model_name)
    print('Model:', model_name, 'Accuracy:', accuracy)
    
if __name__ == '__main__':
    main()