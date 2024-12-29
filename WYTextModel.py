import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import jieba

# 启动cuda，如果能用的话
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建一个在CUDA设备上的生成器
    generator = torch.Generator(device=device)
# 参数预设值
embedding_dim = 256
hidden_dim = 512
batch_size = 1
windows_size = 10


# 简单的LSTM模型
class WYTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(WYTextModel, self).__init__()
        # 嵌入层 （将单词转换为向量）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embdded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embdded)
        hidden = hidden[-1]
        return self.fc(hidden)

# 普通dataset
class WYTextDataset(Dataset):
    def __init__(self, txt_path):
        self.word_to_idx = {'<UNK>': 0}
        self.text_indices = []
        idx = 1
        self.txt_list = open(txt_path, "r", encoding="utf-8").readlines()
        for text in self.txt_list:
            # jieba 分词
            words = jieba.lcut(text)
            # 建立单词索引表
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = idx
                    idx += 1
            # 把每句单词转换成索引数字
            indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
            self.text_indices.append(indices)

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, item):
        # 指定要的句子
        text_seq = self.text_indices[item]
        # 最长句子
        max_length = max([len(seq) for seq in self.text_indices])
        # 填充成最长句子一样长度
        padded_seq = text_seq + [0] * (max_length - len(text_seq))
        # 生成张量
        text_tensor = torch.tensor(padded_seq)
        label_tensor = torch.tensor(self.text_indices[item])
        return text_tensor, label_tensor

    def get_vocab_size(self):
        return len(self.word_to_idx)

# 滑动窗口的文本预测LSTM模型
class WYTextSWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WYTextSWModel, self).__init__()
        # 嵌入层 （将单词转换为向量）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        output = output.view(-1, output.size(-1))
        return output

# 滑动窗口的数据集
class WYTextSWDataset(Dataset):
    def __init__(self, txt_path, slide):
        # 数据加载
        self.file = open(txt_path, "r",encoding="utf-8").read().replace("\r", "").replace("\n", "")
        self.data = jieba.lcut(self.file)
        # 数据预处理
        self.letters = list(set(self.data))
        self.num_letters = len(self.letters)
        # 创建字典
        self.int_to_char = {a: b for a, b in enumerate(self.letters)}
        self.char_to_int = {b: a for a, b in enumerate(self.letters)}
        self.slide = slide
        # 分割后的结果
        text_datas, text_labels = self.extract_data()
        self.text_datas_int, self.text_labels_int = self.char_to_int_data(text_datas, text_labels)

    # 滑动窗口提取数据
    def extract_data(self):
        text_datas, text_labels = [], []
        for i in range(len(self.data) - self.slide):
            text_datas.append([a for a in self.data[i:i + self.slide]])
            text_labels.append(self.data[i + self.slide])
        return text_datas, text_labels

    # 字符转成数字
    def char_to_int_data(self, text_datas, text_labels):
        text_datas_int, text_labels_int = [], []
        for i in range(len(text_labels)):
            text_datas_int.append([self.char_to_int[z] for z in text_datas[i]])
            text_labels_int.append(self.char_to_int[text_labels[i]])
        return text_datas_int, text_labels_int

    def __len__(self):
        return len(self.text_labels_int)

    def __getitem__(self, item):
        data = self.text_datas_int[item]
        label = self.text_labels_int[item]
        return torch.tensor(data), torch.tensor(label)

class TextTrainTools:
    def __init__(self, model, dataloader):
        # 超参数
        self.num_epochs = 10
        self.learning_rate = 0.001
        # 模型
        self.model = model.to(device)
        self.dataloader = dataloader

    # 测试正确率
    def test_accuracy(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        with torch.no_grad():
            for text, label in self.dataloader:
                output = self.model(text.to(device))
                text, predicted_tags = torch.max(output.data, 1)
                # label总数
                label = label.repeat_interleave(windows_size)
                total += label.size(0)
                accuracy += (predicted_tags == label).sum().item()
        accuracy = (100 * accuracy / total)
        return accuracy

    def start_train(self, num_epochs):
        max_accuracy = 80.0
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        # 优化器
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        # 开启训练
        for epoch in tqdm(range(num_epochs)):
            for idx, (text, label) in enumerate(self.dataloader):
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(text.to(device))
                label = label.repeat_interleave(windows_size)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                if idx % 100 == 99:
                    accuracy = self.test_accuracy()
                    print(
                        f'\nEpoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}/{len(self.dataloader)}], Accuracy: {accuracy}\n')
                    if accuracy > max_accuracy:
                        self.save_model()
                        max_accuracy = accuracy
        print(f"训练完毕，最好的正确率：{max_accuracy}")

    def save_model(self):
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"./WYTextModel_{now_time}.pth"
        torch.save(self.model.state_dict(), path)


def history_train():
    dataset = WYTextSWDataset(r"E:\test_model\TextDataset\测试.txt", windows_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    vocab_size = dataset.__len__()
    model = WYTextSWModel(vocab_size, embedding_dim, hidden_dim)
    # 开练
    tt = TextTrainTools(model, dataloader)
    tt.start_train(30)

def history_test():
    dataset = WYTextSWDataset(r"E:\test_model\TextDataset\测试.txt", windows_size)
    test_text, test_label = dataset.__getitem__(2)
    char_ditc = dataset.int_to_char
    print(f"text:{[char_ditc[z.item()] for z in test_text]},label:{[char_ditc[test_label.item()]]}")
    vocab_size = dataset.__len__()
    model = WYTextSWModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(r"E:\test_model\WYTextModel_2024-12-29_23-09-05.pth"))
    model.eval()
    with torch.no_grad():
        output = model(test_text)
        _, predicted = torch.max(output, 1)
        print("Input data shape:", output.shape)
        # _, predicted = torch.max(output, 1)
    print(f"Predicted class: {[char_ditc[z.item()] for z in predicted]},Acc label:{test_label}")

if __name__ == '__main__':
    history_train()
    history_test()