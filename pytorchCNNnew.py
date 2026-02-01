import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import Counter
# dataset.pyから作成したLoaderを読み込む
from dataset import train_loader, test_loader 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデル定義 ---
class MyCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- トレーニング関数 ---
def train():
    model = MyCNN(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    # 学習率を 0.001 に下げて、丁寧に学習させる
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    num_epochs = 10 # データを絞ったので、エポック数は少し増やしてもOK

    print(f"\n学習開始 (Device: {device})")
    print("-" * 40)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_label_counts = Counter()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            epoch_label_counts.update(labels.tolist())

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
        # クラスごとの枚数を確認（すべて同じ数になるはず）
        # print("  学習ラベル分布:", dict(sorted(epoch_label_counts.items())))

    # モデル保存
    os.makedirs("/workspace/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "/workspace/checkpoints/final.pth")#ファイル指定として保存
    print("\n学習完了。モデルを保存しました。")

if __name__ == "__main__":
    train()