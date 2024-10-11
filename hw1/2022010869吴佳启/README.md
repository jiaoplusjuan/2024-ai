1. 文件结构
   Project/
   ├─ README.md
   ├─ src/
   │  ├─ pinyin.py
   │  ├─ refractor_data.py
   │  ├─ graph.py
   │  ├─ test.py
   │  └─ control.sh
   ├─ data/
   │  ├─ input.txt
   │  └─ output.txt
2. 文件功能解释与运行方法

- refactor_data.py: 实现从语料库中提取中间材料，可以提取一元、二元和三元语料，命令参考如下：

```
python refractor_data.py 1 ../data/2016-04.txt ../data/2016-05.txt
```

其中第二参数为提取语料方法，1表示提取到二元，2表示提取到三元语料，3，4表示通过逐句划分获得二元、三元语料，后面跟的参数为语料的路径，推荐把所有语料都放置在data文件夹下，同时需要注意，**要先把“拼音汉字表.txt”放在src路径下，并且文件的编码形式都要为"utf_8"**。
如果此时处理正确，你目前的文件结构应该如下所示：  
  Project/
   ├─ README.md
   ├─ src/
   │  ├─ pinyin.py
   │  ├─ refractor_data.py
   │  ├─ graph.py
   │  ├─ test.py
   │  ├─ 1_word.txt
   │  ├─ 2_word.txt
   │  ├─ 3_word.txt
   │  ├─ word2pinyin.txt
   │  ├─ 拼音汉字表.txt
   │  └─ control.sh
   ├─ data/
   │  ├─ input.txt
   │  ├─ output.txt
   │  └─ std_output.txt

- pinyin.py: 实现汉字转拼音功能，其中蕴含二元模型和三元模型，请注意每次运行会进行一次句准确率的检测，**因此需要提前把"std_output.txt"放在data文件夹下，编码为utf_8**。命令参考如下：

```
python pinyin.py ../data/input.txt ../data/output.txt 0.9 0 1
python pinyin.py ../data/input.txt ../data/output.txt 0.9 0.9 2
```

其中第二、三参数为输入输出文件路径，第四、五参数为二元模型和三元模型的参数值，即LAMBDA和ALPHA，第六参数为选择使用哪种模型，1表示二元模型，2表示三元模型。**参数选择不可为1，应该为0-1的小数，否则可能出现错误！**
运行后会输出一个句准确率。
**请注意，一定要保证训练的语料充足，否则可能出现某些读音在运行时没有对应的字而出现报错！**

- complete.sh 脚本文件，用于三元时绘制曲线图，需要先使用refractor_data文件获取三元数据，然后运行如下命令：

```
./complete.sh 
```

运行后会将数据写入results.txt文件里

- graph.py: 通过results.txt实现绘制准确率曲线图功能，命令参考如下：

```
python graph.py 
```

- test.py
  用于完成pinyin.py后实现output.txt与std_ouput.txt的对比，输出字句的准确率，命令参考如下：

```
python test.py 
```

3. 可能需要安装的库：

- jieba
- matplotlib
- numpy
- tqdm
- pypinyin
  可以通过pip安装，命令如下：

```
pip install jieba matplotlib numpy tqdm pypinyin
```
