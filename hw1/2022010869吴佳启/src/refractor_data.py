from collections import Counter
import re
import json
from collections import defaultdict
from tqdm import tqdm
from pypinyin import lazy_pinyin
import jieba
import sys
import time
pinyin_dict = {}
text = ""
chinese_bigrams = defaultdict(int)
chinese_trigrams = defaultdict(int)
chinese_unigrams = defaultdict(int)
chinese_first_unigrams = defaultdict(int)

def init():
    # 建立字到拼音的映射
    global pinyin_dict
    with open('拼音汉字表.txt', 'r',encoding='utf-8') as source_file:
        lines = source_file.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        pinyin = parts[0]
        characters = parts[1:]
        for char in characters:
            if char in pinyin_dict:
                pinyin_dict[char].append(pinyin)
            else:
                pinyin_dict[char] = [pinyin]
    
    with open('word2pinyin.txt', 'w', encoding='utf-8') as output_file:
        for char, pinyins in pinyin_dict.items():
            output_file.write(f"{char}: {', '.join(pinyins)}\n")
        
def query_pinyin(char):
    if char in pinyin_dict:
        return pinyin_dict[char]
    else:
        return "0"
# 将整体当做一个句子读入
def read_as_a_big_sentence(file_paths):
    global text
    # 使用 tqdm 显示读取文件的进度条
    with tqdm(total=len(file_paths), desc="TASK1: Reading files") as pbar:
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                text += file.read()    
            pbar.update(1)
    chinese_words = re.findall(r'[\u4e00-\u9fa5]+', text)
    chinese_text = ''.join(chinese_words)
    with tqdm(total=len(chinese_text), desc="TASK2: Processing sentences") as pbar:
        for i in range(len(chinese_text) - 2):
            bigram = chinese_text[i] + ' ' + chinese_text[i + 1]
            chinese_bigrams[bigram] += 1
            unigram = chinese_text[i]
            if i == 0:
                chinese_first_unigrams[unigram] += 1
            chinese_unigrams[unigram] += 1
            trigram = chinese_text[i] + ' ' + chinese_text[i + 1] + ' ' + chinese_text[i + 2]
            chinese_trigrams[trigram] += 1
            pbar.update(1)
            
# 分成一个一个句子读入            
def read_by_sentences(file_paths):
    global text
    # 使用 tqdm 显示读取文件的进度条
    with tqdm(total=len(file_paths), desc="TASK1: Reading files") as pbar:
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                text += file.read()    
            pbar.update(1)
    chinese_words = re.findall(r'[\u4e00-\u9fa5]+', text)
    with tqdm(total=len(chinese_words), desc="TASK2: Processing sentences") as pbar:
        for word in chinese_words:
            for i in range(len(word)):
                unigram = word[i]
                if i == 0:
                    chinese_first_unigrams[unigram] += 1
                chinese_unigrams[unigram] += 1
                if i + 1 < len(word):
                    bigram = word[i] + ' ' + word[i + 1]
                    chinese_bigrams[bigram] += 1
                if i + 2 < len(word):
                    trigram = word[i] + ' ' + word[i + 1] + ' ' + word[i + 2]
                    chinese_trigrams[trigram] += 1
            pbar.update(1)

def task_get_bigrams(file_paths):
    read_as_a_big_sentence(file_paths)
    write(1)

def task_get_bigrams_by_sentence(file_paths):
    read_by_sentences(file_paths)
    write(1)
        
def task_get_trigrams(file_paths):
    read_as_a_big_sentence(file_paths)
    write(2)

def task_get_trigrams_by_sentence(file_paths):
    read_by_sentences(file_paths)
    write(2)

def write(method):
    # 处理句首字的部分
    output_data_unigram = {}
    for unigram, count in tqdm(chinese_first_unigrams.items(), desc='Processing first unigrams'):
        pinyins = query_pinyin(unigram[0])
        for pinyin in pinyins:
            if pinyin == "0":
                continue
            pinyin = f"{pinyin}"
            if pinyin not in output_data_unigram:
                output_data_unigram[pinyin] = {"words": [], "counts": []}
            output_data_unigram[pinyin]["words"].append(unigram)
            output_data_unigram[pinyin]["counts"].append(count)
    # 处理一元的部分
    output_data_unigram = {}
    for unigram, count in tqdm(chinese_unigrams.items(), desc='Processing unigrams'):
        pinyins = query_pinyin(unigram[0])
        for pinyin in pinyins:
            if pinyin == "0":
                continue
            pinyin = f"{pinyin}"
            if pinyin not in output_data_unigram:
                output_data_unigram[pinyin] = {"words": [], "counts": []}
            output_data_unigram[pinyin]["words"].append(unigram)
            output_data_unigram[pinyin]["counts"].append(count)

    with open('1_word.txt', 'w', encoding='utf-8') as output_file:
        json.dump(output_data_unigram, output_file, ensure_ascii=False, indent=2)
        
    # 处理二元的部分
    output_data_bigram = {}
    for bigram, count in tqdm(chinese_bigrams.items(), desc='Processing bigrams'):
        pinyins1 = query_pinyin(bigram[0])
        pinyins2 = query_pinyin(bigram[2])
        # 先通过lazy拼音查找这个拼音是否在组合里，减少
        test_pinyin = lazy_pinyin(bigram[0]+bigram[2])
        test_pinyin1 = test_pinyin[0]
        test_pinyin2 = test_pinyin[1]
        if test_pinyin1 in pinyins1 and test_pinyin2 in pinyins2:
            pinyin_pair = f"{test_pinyin1} {test_pinyin2}"
            if pinyin_pair not in output_data_bigram:
                output_data_bigram[pinyin_pair] = {"words": [], "counts": []}
            output_data_bigram[pinyin_pair]["words"].append(bigram)
            output_data_bigram[pinyin_pair]["counts"].append(count)
        else :
            for pinyin1 in pinyins1:
                for pinyin2 in pinyins2:
                    if pinyin1 == "0" or pinyin2 == "0":
                        continue
                    pinyin_pair = f"{pinyin1} {pinyin2}"
                    if pinyin_pair not in output_data_bigram:
                        output_data_bigram[pinyin_pair] = {"words": [], "counts": []}
                    output_data_bigram[pinyin_pair]["words"].append(bigram)
                    output_data_bigram[pinyin_pair]["counts"].append(count)

    with open('2_word.txt', 'w', encoding='utf-8') as output_file:
        json.dump(output_data_bigram, output_file, ensure_ascii=False, indent=2)
    if method == 2:
        # 处理三元的部分
        output_data_trigram = {}
        for trigram, count in tqdm(chinese_trigrams.items(), desc='Processing trigrams'):
            pinyins1 = query_pinyin(trigram[0])
            pinyins2 = query_pinyin(trigram[2])
            pinyins3 = query_pinyin(trigram[4])
            test_pinyin = lazy_pinyin(trigram[0]+trigram[2]+trigram[4])
            test_pinyin1 = test_pinyin[0]
            test_pinyin2 = test_pinyin[1]
            test_pinyin3 = test_pinyin[2]
            if test_pinyin1 in pinyins1 and test_pinyin2 in pinyins2 and test_pinyin3 in pinyins3:
                pinyin_pair = f"{test_pinyin1} {test_pinyin2} {test_pinyin3}"
                if pinyin_pair not in output_data_trigram:
                    output_data_trigram[pinyin_pair] = {"words": [], "counts": []}
                output_data_trigram[pinyin_pair]["words"].append(trigram)
                output_data_trigram[pinyin_pair]["counts"].append(count)
            else:
                for pinyin1 in pinyins1:
                    for pinyin2 in pinyins2:
                        for pinyin3 in pinyins3:
                            if pinyin1 == "0" or pinyin2 == "0" or pinyin3 == "0":
                                continue
                            pinyin_pair = f"{pinyin1} {pinyin2} {pinyin3}"
                            if pinyin_pair not in output_data_trigram:
                                output_data_trigram[pinyin_pair] = {"words": [], "counts": []}
                            output_data_trigram[pinyin_pair]["words"].append(trigram)
                            output_data_trigram[pinyin_pair]["counts"].append(count)

        with open('3_word.txt', 'w', encoding='utf-8') as output_file:
            json.dump(output_data_trigram, output_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    file_paths = []
    start_time = time.perf_counter()
    task = int(sys.argv[1])
    for arg in sys.argv[2:]:  # 从第二个参数开始遍历
        file_paths.append(arg)
    init()
    if task == 1:
        task_get_bigrams(file_paths)
    elif task == 2:
        task_get_trigrams(file_paths)
    elif task == 3:
        task_get_bigrams_by_sentence(file_paths)
    elif task == 4:
        task_get_trigrams_by_sentence(file_paths)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")