

import json
import torch
import torch.nn as nn
import torch_npu
import timm
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

# ==== 配置参数 ====
input_jsonl = "/root/zhangliang/Qwen2.5-VL/trydata/all_chaifen_cc.jsonl"
output_jsonl = "/root/zhangliang/Qwen2.5-VL/yanke/infer_tagged_data.jsonl"
model_path = "/root/zhangliang/Qwen2.5-VL/yanke/best_model.pth"
device = torch.device("npu:0")

label_list = [
    "窥不入", "影动", "黄斑区", "中心凹", "视盘", "动脉", "静脉", "毛细血管", "网膜下遮蔽", "后极部",
    "中周部", "眼周边", "拱环", "高荧光", "低荧光", "荧光着染", "遮蔽荧光", "积存", "色素上皮",
    "血管壁染", "荧光渗漏", "透见荧光", "点状", "片状", "簇样", "点片状", "微血管瘤样", "不规则",
    "弥漫样", "息肉样", "变性区", "异常血管", "萎缩灶", "激光斑", "出血", "呈高度近视眼底改变",
    "网膜漂浮", "NP", "NV", "NVD"
]
threshold = 0.5
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==== 加载模型 ====
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(label_list))
model.classifier = nn.Linear(model.classifier.in_features, len(label_list))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ==== 图像预测函数 ====
def infer_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.sigmoid(output).squeeze().cpu()
    tags = [label_list[i] for i, p in enumerate(probs) if p > threshold]
    return tags

# ==== 推理所有数据 ====
with open(input_jsonl, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

with open(input_jsonl, 'r', encoding='utf-8') as fin, open(output_jsonl, 'w', encoding='utf-8') as fout:
    for line in tqdm(fin, desc="推理中", total=total_lines):
        entry = json.loads(line)
        found = False
        for msg in entry["messages"]:
            if msg.get("role") == "user" and "image" in msg:
                img_path = msg["image"]
                tags = infer_image(img_path)
                entry["tag"] = tags
                found = True
                break
        if not found:
            entry["tag"] = []
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
