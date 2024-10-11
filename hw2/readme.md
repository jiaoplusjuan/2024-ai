1. 代码结构：

需要将Dataset文件放在目录下，得到最后的结构如下：
```
├── Dataset
│   ├── test.txt
│   ├── train.txt
│   ├── validation.txt
│   ├── wiki_word2vec_50.bin
├── bert-base-chinese
├── dataLoder.py
├── model.py
├── main.py
├── photo.py
├── README.md
```

2. 文件解析：
- Dataset文件夹下存放了训练集，验证集和测试集的数据文件，还有bin文件，为该实验提供的数据
- dataLoder.py文件中定义了数据加载器，该加载器读取数据文件，并将其转换为PyTorch可以处理的格式。
- models.py文件中定义了各种模型。
- main.py文件中定义了训练和测试的流程，并使用数据加载器加载数据，利用模型完成训练。
- photo.py利用了matplotlib库，绘制了训练过程中的损失和准确率的变化。
- bert-base-chinese文件夹下存放了BERT模型的预训练文件。
3. 代码运行：

其中指定了使用GPU，使用模型的类型，learning—rate，batch_size，epochs等参数，具体运行命令可以参考如下。
```
python main.py --gpu --model TextCNN --learning-rate 0.001 --batch-size 64 --epochs 20
python main.py --gpu --model RNN_GRU --learning-rate 0.001 --batch-size 64 --epochs 30
python main.py --gpu --model RNN_LSTM --learning-rate 0.001 --batch-size 64 --epochs 30
python main.py --gpu --model MLP --learning-rate 0.001 --batch-size 64 --epochs 20
python main.py --gpu --model Transformer --learning-rate 0.001 --batch-size 64 --epochs 20
python main.py --gpu --model BERT --learning-rate 0.00001 --batch-size 50 --epochs 20
python main.py --gpu --model BIRNN --learning-rate 0.001 --batch-size 64 --epochs 20
python main.py --gpu --model ALL --learning-rate 0.001 --batch-size 64 --epochs 5
```
4. 注意事项
- 在运行BERT模型时，如果设备没有gpu会耗时极长，同时注意内存，若内存不够时可以适当调小运行BERT的batch_size
