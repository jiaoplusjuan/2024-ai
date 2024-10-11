import sys
import json
import math
from tqdm import tqdm
import time
data1 = {}
data2 = {}
data3 = {}
CONST_LAMBDA = 0.87
CONST_ALPHA = 0
CONST_ALL_COUNT = 0
file1_path = "../data/output.txt"
file2_path = "../data/std_output.txt"
def load_data_from_1_word():
    global data1 
    with open('./1_word.txt', 'r', encoding='utf-8') as file:
        data1 = json.load(file)

def load_data_from_2_word():
    global data2
    with open('./2_word.txt', 'r', encoding='utf-8') as file:
        data = json.load(file)
    data2 = {}
    for key, value in data.items():
        word_counts = {}
        for i in range(len(value['words'])):
            word = value['words'][i].replace(' ', '')  # 去掉空格
            word_counts[word] = value['counts'][i]

        data2[key] = word_counts  # 将 word_counts 直接赋值给 data2[key] 
        
def load_data_from_3_word():
    global data3
    with open('./3_word.txt', 'r', encoding='utf-8') as file:
        data = json.load(file)
    data3 = {}
    for key, value in tqdm(data.items(), desc='Processing data', total=len(data)):
        word_counts = {}
        for i in range(len(value['words'])):
            word = value['words'][i].replace(' ', '')  # 去掉空格
            word_counts[word] = value['counts'][i]

        data3[key] = word_counts  # 将 word_counts 直接赋值给 data2[key] 

class Point:
    character: str # 记录存储的汉字
    cost_all: float # 从初始点开始到达他的代价
    front: 'Point'
    
    def __init__(self, character: str):
        self.character = character
        self.cost_all = 10000000
        self.front = None

def get_all_count():
    global CONST_ALL_COUNT
    for key, value in data1.items():
        CONST_ALL_COUNT += sum(value['counts'])
    # print(CONST_ALL_COUNT)

# 获取一元概率
def get_unigram_probabilty(pinyin, index):
    count = data1[pinyin]["counts"][index]
    return count/CONST_ALL_COUNT

# 获取二元概率
def get_bigram_probability(pinyin_before, character_before_index,pinyin_together, character_together):
    frequency_together = data2.get(pinyin_together, {}).get(character_together, 0)
    frequency_before_word = data1[pinyin_before]["counts"][character_before_index]
    conditional_probability = frequency_together/frequency_before_word
    return conditional_probability

# 获取三元概率
def get_trigram_probability(character01, pinyin01,character012,pinyin012):
    frequency01 = data2.get(pinyin01, {}).get(character01, 1)
    frequency012 = data3.get(pinyin012, {}).get(character012, 0)
    conditional_probability = frequency012/frequency01
    return conditional_probability

# 计算cost
def cost(unigram_probabilty, bigram_probabilty = -1, trigram_probabilty = -1):
    if bigram_probabilty == -1 and trigram_probabilty == -1:
        return -math.log(unigram_probabilty)
    elif bigram_probabilty != -1 and trigram_probabilty == -1:
        return -math.log(CONST_LAMBDA * bigram_probabilty + (1 - CONST_LAMBDA) * unigram_probabilty)
    else:
        return -math.log(CONST_ALPHA * trigram_probabilty + (1 - CONST_ALPHA) * (CONST_LAMBDA * bigram_probabilty + (1 - CONST_LAMBDA) * unigram_probabilty))

# 三元模型
def model2_get_character(line):
    pinyins = line.split()
    first_pinyin = True
    second_pinyin = True
    matrix_rows = len(pinyins) + 1
    matrix = [[] for _ in range(matrix_rows)]
    matrix[0].append(Point(""))  # 初始点
    rows_now = 2
    for pinyin in pinyins:
        if first_pinyin:
            for i in range(len(data1[pinyin]["words"])):
                matrix[1].append(Point(data1[pinyin]["words"][i]))
                matrix[1][i].cost_all = -math.log(data1[pinyin]["counts"][i]/CONST_ALL_COUNT) # 第一个字符特殊处理，即这个字出现在这个拼音里的概率
                matrix[1][i].front = matrix[0][0]
            first_pinyin = False
        elif second_pinyin: # 特殊处理第二列的情况
            for i in range(len(data1[pinyin]["words"])):
                matrix[rows_now].append(Point(data1[pinyin]["words"][i])) # 加入新的数据
            pinyin_now = pinyins[rows_now - 1]
            pinyin_before = pinyins[rows_now - 2]
            pinyin_together =  pinyin_before + " " + pinyin_now
            for i in range(len(matrix[rows_now])):
                for j in range(len(matrix[rows_now-1])):
                    character_together = matrix[rows_now-1][j].character + matrix[rows_now][i].character
                    unigram_probabilty = get_unigram_probabilty(pinyin_now, i)
                    biagram_probabilty = get_bigram_probability(pinyin_before, j, pinyin_together, character_together)
                    local_cost = cost(unigram_probabilty, biagram_probabilty)
                    if local_cost + matrix[rows_now-1][j].cost_all < matrix[rows_now][i].cost_all:
                        matrix[rows_now][i].cost_all = local_cost + matrix[rows_now-1][j].cost_all
                        matrix[rows_now][i].front = matrix[rows_now-1][j]
            rows_now += 1
            second_pinyin = False
        else:
            for i in range(len(data1[pinyin]["words"])):
                matrix[rows_now].append(Point(data1[pinyin]["words"][i])) # 加入新的数据
            pinyin2 = pinyins[rows_now-1]
            pinyin1 = pinyins[rows_now-2]
            pinyin12 = pinyin1 + " " + pinyin2
            pinyin0 = pinyins[rows_now-3]
            pinyin01 = pinyin0 + " " + pinyin1
            pinyin012 = pinyin0 + " " + pinyin1 + " " + pinyin2
            for i in range(len(matrix[rows_now])):
                for j in range(len(matrix[rows_now-1])):
                    matrix_front = matrix[rows_now-1][j].front
                    character12 =matrix[rows_now-1][j].character + matrix[rows_now][i].character
                    character012 = matrix_front.character + matrix[rows_now-1][j].character + matrix[rows_now][i].character
                    character01 = matrix_front.character + matrix[rows_now-1][j].character
                    unigram_probabilty = get_unigram_probabilty(pinyin2, i)
                    biagram_probabilty = get_bigram_probability(pinyin1, j, pinyin12, character12)
                    trigram_probability = get_trigram_probability(character01,pinyin01,character012,pinyin012)
                    local_cost = cost(unigram_probabilty, biagram_probabilty, trigram_probability)
                    if local_cost + matrix[rows_now-1][j].cost_all < matrix[rows_now][i].cost_all:
                        matrix[rows_now][i].cost_all = local_cost + matrix[rows_now-1][j].cost_all
                        matrix[rows_now][i].front = matrix[rows_now-1][j]
            rows_now += 1 
    select_position = 0
    min_cost = matrix[rows_now-1][0].cost_all
    for i in range(len(matrix[rows_now-1])):
        if(matrix[rows_now-1][i].cost_all < min_cost):
            select_position = i
            min_cost = matrix[rows_now-1][i].cost_all
    output = ""
    point_now = matrix[rows_now-1][select_position]
    for i in range(len(pinyins)):
        output += point_now.character
        point_now = point_now.front
    return output[::-1]

def model1_get_character(line):
    pinyins = line.split()
    first_pinyin = True
    matrix_rows = len(pinyins) + 1
    matrix = [[] for _ in range(matrix_rows)]
    matrix[0].append(Point(""))  # 初始点
    rows_now = 2
    for pinyin in pinyins:
        if first_pinyin:
            for i in range(len(data1[pinyin]["words"])):
                matrix[1].append(Point(data1[pinyin]["words"][i]))
                matrix[1][i].cost_all = -math.log(data1[pinyin]["counts"][i]/CONST_ALL_COUNT) # 第一个字符特殊处理，即这个字出现在这个拼音里的概率
                matrix[1][i].front = matrix[0][0]
            first_pinyin = False
        else:
            for i in range(len(data1[pinyin]["words"])):
                matrix[rows_now].append(Point(data1[pinyin]["words"][i])) # 加入新的数据
            pinyin_before = pinyins[rows_now - 2]
            pinyin_now = pinyins[rows_now - 1]
            pinyin_together =  pinyin_before + " " + pinyin_now
            for i in range(len(matrix[rows_now])):
                for j in range(len(matrix[rows_now-1])):
                    character_together = matrix[rows_now-1][j].character + matrix[rows_now][i].character
                    local_probabilty = get_unigram_probabilty(pinyin_now, i)
                    biagram_probabilty = get_bigram_probability(pinyin_before, j, pinyin_together, character_together)
                    local_cost = cost(local_probabilty, biagram_probabilty)
                    if local_cost + matrix[rows_now-1][j].cost_all < matrix[rows_now][i].cost_all:
                        matrix[rows_now][i].cost_all = local_cost + matrix[rows_now-1][j].cost_all
                        matrix[rows_now][i].front = matrix[rows_now-1][j]
            rows_now += 1
    select_position = 0
    min_cost = matrix[rows_now-1][0].cost_all
    for i in range(len(matrix[rows_now-1])):
        if(matrix[rows_now-1][i].cost_all < min_cost):
            select_position = i
            min_cost = matrix[rows_now-1][i].cost_all
    output = ""
    point_now = matrix[rows_now-1][select_position]
    for i in range(len(pinyins)):
        output += point_now.character
        point_now = point_now.front
    return output[::-1]

def init(type):
    load_data_from_1_word()
    load_data_from_2_word()
    if type == 2:
        load_data_from_3_word()
    get_all_count()

def compare_files(file1, file2, encoding='utf-8'):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding=encoding) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        # 行级别的正确率
        num_lines = min(len(lines1), len(lines2))
        correct_lines = sum(1 for line1, line2 in zip(lines1, lines2) if line1.strip() == line2.strip())
        line_accuracy = (correct_lines / num_lines) * 100
        print(f"{line_accuracy:.2f}")
    

def main(input_file_path, output_file_path, coefficient1, coefficient2, type):
    # 通过参数设置全局变量的大小
    global CONST_LAMBDA
    global CONST_ALPHA
    CONST_LAMBDA = coefficient1
    CONST_ALPHA = coefficient2
    # 打开输入文件并读取数据
    init(type)
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        input_data = input_file.readlines()
    
    results = []
    # 使用 Viterbi 算法处理每一行数据
    for line in tqdm(input_data, desc='Processing data'):
        if type == 1:
            result = model1_get_character(line)
        else: 
            result = model2_get_character(line)
        results.append(result)
    
    # 打开输出文件并将结果写入
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in results:
            output_file.write(line + "\n")
    
    compare_files(file1_path, file2_path)

def check(coefficient1, coefficient2):
    if not isinstance(coefficient1, float):
        print("The third argument should be a float value.")
        sys.exit(1)
    if not (0 <= coefficient1 < 1.0):
        print("The coefficient should be between 0 and 1.0.")
        sys.exit(1)    
    if not isinstance(coefficient2, float):
        print("The third argument should be a float value.")
        sys.exit(1)
    if not (0 <= coefficient2 < 1.0):
        print("The coefficient should be between 0 and 1.0.")
        sys.exit(1)   
          
if __name__ == "__main__":
    start_time = time.perf_counter()
    if len(sys.argv) != 6:
        print("Usage: python pinyin.py input_file output_file")
        sys.exit(1)
    # 获取输入输出文件路径
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    # 获取两个参数
    coefficient1 = float(sys.argv[3])
    coefficient2 = float(sys.argv[4])
    check(coefficient1,coefficient2)
    type = int(sys.argv[5])
    main(input_file_path, output_file_path,coefficient1, coefficient2, type)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # 计算运行时间
    print("Elapsed time:", elapsed_time, "seconds")
    sys.exit(0)
