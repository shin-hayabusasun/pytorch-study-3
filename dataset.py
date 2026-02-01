import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

# 1. データセットのロード
DATASET_NAME = "prithivMLmods/Face-Age-10K"
raw_dataset = load_dataset(DATASET_NAME)
df = pd.DataFrame(raw_dataset['train'])

# 2. 画像をNumPy配列に変換（後でPILに戻してTransformを適用するため）
df['image'] = df['image'].apply(lambda img: np.array(img))

# 3. データセットの数を最小クラスに統一（バランス調整）   ####ここが勉強用と違う####
min_count = df['label'].value_counts().min()
print(f"各クラスを {min_count} 枚に統一してバランスを整えます。")
df = df.groupby('label').sample(n=min_count, random_state=42).reset_index(drop=True)

# 4. 学習用とテスト用に分割 (8:2)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

# 5. Datasetクラスの定義
class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['label']
        
        # NumPy配列をPILに戻す
        image = Image.fromarray(image).convert("RGB")     ####ここが勉強用と違う####
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# 6. Transformの定義（推論時のload.pyと条件を合わせる）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) # これが推論時と同じであることが重要   ####ここが勉強用と違う####
])

# 7. DataLoaderの作成
train_dataset = MyDataset(train_df, transform=transform)
test_dataset = MyDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print(f"学習データ数: {len(train_dataset)}")
    print("ラベル分布:\n", train_df['label'].value_counts())