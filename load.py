import torch
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルロード
# 1. モデルを作成
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=8)

# 2. 重みをロード（map_locationを指定しても、念のため次に進む）
state_dict = torch.load("/workspace/checkpoints/final.pth", map_location=device)
model.load_state_dict(state_dict)

# 3. 【重要】「モデル全体」を確実にGPUへ送る
model.to(device) 

# 4. 推論モードに
model.eval()

# 画像取得
url = "https://pigeon.info/sc_rsc/baby/images/theme/room/test/m_pic03.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")
url1 = "https://assets-www.leon.jp/image/2019/05/01100310880420/1600/190422_U1Q4033.jpg"
response1 = requests.get(url1)
img1 = Image.open(BytesIO(response1.content)).convert("RGB")
url2 = "https://scalp-d.angfa-store.jp/brand/dism/column/30s-acne/image/01.jpg"
response2 = requests.get(url2)
img2 = Image.open(BytesIO(response2.content)).convert("RGB")

# 変換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
    
x = transform(img).unsqueeze(0).to(device)#トランスフォーム適応
x1 = transform(img1).unsqueeze(0).to(device)
x2 = transform(img2).unsqueeze(0).to(device)

# 推論
with torch.no_grad():
    outputs = model(x)
    outputs1 = model(x1)
    outputs2 = model(x2)
pred_class = outputs.argmax(dim=1)
pred_class1 = outputs1.argmax(dim=1)
pred_class2 = outputs2.argmax(dim=1)
print("予測クラス:", pred_class.item())
print("予測クラス1:", pred_class1.item())
print("予測クラス2:", pred_class2.item())