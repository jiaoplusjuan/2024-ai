#!/bin/bash

# 定义 x 和 y 的范围
x_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)
y_values=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)

# 创建一个文件来保存结果
result_file="results.txt"

# 清空结果文件
> $result_file

# 计算总迭代次数
total_iterations=$(( ${#x_values[@]} * ${#y_values[@]} ))

# 迭代计算结果并将结果写入文件
current_iteration=0
for x in "${x_values[@]}"; do
    for y in "${y_values[@]}"; do
        result=$(python pinyin2.py ../data/input.txt ../data/output.txt $x $y 2)
        echo "$x $y $result" >> $result_file
        current_iteration=$((current_iteration + 1))
        progress=$(echo "scale=2; $current_iteration * 100 / $total_iterations" | bc)
        echo -ne "Progress: $progress%\\r"
    done
done

echo -ne "\n"
