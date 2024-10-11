import matplotlib.pyplot as plt

# 定义绘制训练损失图的函数
def plot_losses_train(losses_dict):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10.colors  # 使用内置的颜色循环
    for i, (model_name, losses) in enumerate(losses_dict.items()):
        plt.plot(losses, label=model_name, color=colors[i % len(colors)])  # 循环使用颜色
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('train_losses.png')
    
# 定义绘制训练损失图的函数
def plot_losses_val(losses_dict):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10.colors  # 使用内置的颜色循环
    for i, (model_name, losses) in enumerate(losses_dict.items()):
        plt.plot(losses, label=model_name, color=colors[i % len(colors)])  # 循环使用颜色
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('val_losses.png')

# 定义绘制测试准确率图的函数
def plot_test_accuracies(test_accuracies):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    for i, (model_name, acc) in enumerate(test_accuracies.items()):
        plt.plot(acc, label=model_name, color=colors[i % len(colors)])
    plt.title('Test Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('test_accuracies.png')

# 定义绘制验证准确率图的函数
def plot_val_accuracies(val_accuracies):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    for i, (model_name, acc) in enumerate(val_accuracies.items()):
        plt.plot(acc, label=model_name, color=colors[i % len(colors)])
    plt.title('Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('val_accuracies.png')

# 定义绘制训练准确率图的函数
def plot_train_accuracies(train_accuracies):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    for i, (model_name, acc) in enumerate(train_accuracies.items()):
        plt.plot(acc, label=model_name, color=colors[i % len(colors)])
    plt.title('Train Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('train_accuracies.png')
    
# 定义绘制训练准确率图的函数
def plot_f1_scores(f1_scores):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    for i, (model_name, f1_score) in enumerate(f1_scores.items()):
        plt.plot(f1_score, label=model_name, color=colors[i % len(colors)])
    plt.title('F1 Scores')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Scores')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('f1_scores.png')
    
import matplotlib.pyplot as plt

def plot_model(train_losses, val_losses, train_accuracies, test_accuracies, val_accuracies, f1_scores, model_name):
    # 设置图表的大小
    plt.figure(figsize=(14, 20))  # 调整高度以适应三行的布局

    # 绘制损失 - 第一个子图 (1行3列的第一个)
    plt.subplot(3, 3, 1)
    plt.plot(train_losses, label='Train Losses')
    plt.plot(val_losses, label='Validation Losses')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率 - 第二个子图 (1行3列的第二个)
    plt.subplot(3, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracies')
    plt.plot(test_accuracies, label='Test Accuracies')
    plt.plot(val_accuracies, label='Validation Accuracies')
    plt.title('Training, Test and Validation Accuracies Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制 F1 分数 - 第三个子图 (1行3列的第三个)
    plt.subplot(3, 3, 3)
    plt.plot(f1_scores, label='F1 Scores')
    plt.title('F1 Scores Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # 调整子图间距
    plt.tight_layout()

    # 保存图表到文件系统
    plt.savefig('model'+model_name + '.png', dpi=300, bbox_inches='tight')

    # 关闭图表，释放内存
    plt.close()