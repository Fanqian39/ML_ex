import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

# 假设我们有一个训练好的模型和测试数据集
model.eval()  # 设置模型为评估模式

# 假设 test_loader 是我们的测试数据集的 DataLoader
all_preds = []
all_labels = []

with torch.no_grad():  # 禁用梯度计算
    for data, labels in test_loader:
        outputs = model(data)  # 前向传播得到输出
        _, preds = torch.max(outputs, 1)  # 获取预测值
        all_preds.extend(preds.numpy())  # 将预测值添加到列表
        all_labels.extend(labels.numpy())  # 将真实标签添加到列表

# 计算评估指标
accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'准确率: {accuracy:.2f}')
print(f'精确率: {precision:.2f}')
print(f'召回率: {recall:.2f}')
print(f'F1分数: {f1:.2f}')