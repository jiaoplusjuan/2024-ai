def compare_files(file1, file2, encoding='utf-8'):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding=encoding) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        # 行级别的正确率
        num_lines = min(len(lines1), len(lines2))
        correct_lines = sum(1 for line1, line2 in zip(lines1, lines2) if line1.strip() == line2.strip())
        line_accuracy = (correct_lines / num_lines) * 100

        # 字级别的正确率
        text1 = ''.join(lines1)
        text2 = ''.join(lines2)
        num_chars = max(len(text1), len(text2))
        correct_chars = sum(1 for char1, char2 in zip(text1, text2) if char1 == char2)
        char_accuracy = (correct_chars / num_chars) * 100

        return line_accuracy, char_accuracy
file1_path = "../data/output.txt"
file2_path = "../data/std_output.txt"
line_accuracy, char_accuracy = compare_files(file1_path, file2_path, encoding='utf-8')
print(f"Line accuracy: {line_accuracy:.2f}%")
print(f"Character accuracy: {char_accuracy:.2f}%")