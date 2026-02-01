import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# 上記で定義したMyDataset, train_loader等は定義済みとします
from dataset import train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. モデルのロード (例: ViT Tiny)
# num_classes=8 を指定すると、分類器(head)が自動的に作成されます pretrained=Trueで事前学習済みモデルを使用 num_classes=8でリセット
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=8)
model = model.to(device)

# 2. パラメータの固定 (Freeze)
# まず、すべてのパラメータを学習対象外にする
for param in model.parameters():
    param.requires_grad = False

# 3. 特定の層だけを学習対象にする (Unfreeze)
# 分類器 (head) のパラメータを学習対象にする
for param in model.head.parameters():
    param.requires_grad = True

# 自己注意 (Self-Attention) の最後のブロックだけを学習対象にする
# ViTの場合、model.blocks[-1] が最後の層に相当します
for param in model.blocks[-1].parameters():
    param.requires_grad = True

# 4. 学習準備
criterion = nn.CrossEntropyLoss()
# 学習対象 (requires_grad=True) のパラメータのみをOptimizerに渡す
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# 学習ループの例
def train_timm():
    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, labels in train_loader:
            # ViTは通常 224x224 入力のため、リサイズが必要な場合はTransformで調整
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "/workspace/checkpoints/final.pth")

if __name__ == "__main__":
    train_timm()