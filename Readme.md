# 👁️ AAC-VLM: Agent-Anchored Collaborative Framework for Multi-Image Clinical Reports

AAC-VLM 是一个专为医疗多图辅助诊断（如眼底荧光血管造影 FFA）设计的智能体锚定协作框架。我们通过引入具备眼科专业知识的专科智能体 (EvidenceAnchorAgent, EAA) 作为视觉语义锚点，将图像理解与格式化输出彻底解耦，有效克服了视觉语言模型 (VLM) 在强格式约束下常见的“格式诱导幻觉”。

## 📑 目录

- [🌟 核心亮点]
- [🏗️ 框架概览]
- [🛠️ 环境准备]
- [📂 数据准备]
- [📦 模型权重]
- [🚀 快速开始]
- [📝 完整流程]
- [🤝 致谢 ]

## 🌟 核心亮点

智能体锚定协作：首次引入轻量级专科智能体 EAA，在多图工作流中重建可靠的视觉证据链。

缓解格式诱导幻觉：通过“两阶段结构化提示”和“逐帧生成-跨帧汇总”策略，彻底解耦读图与报告书写。

临床级可靠性：在真实 FFA 场景测试中，70.18% 的生成报告在专家盲评中被评为具备直接临床可用性。

## 🏗️ 框架概览

> AAC-VLM 包含三个核心组件：
(1) Agent 辅助的句子级过滤；

(2) 两阶段结构化提示构建；

(3) 基于基座 VLM 的两阶段报告生成与跨帧汇总策略。

## 🛠️ 环境准备

本项目基于 Ascend 910 NPU 开发并支持 PyTorch 框架。

### 1) 克隆仓库

```bash
git clone https://github.com/SeuZL/Agent-Anchored_Collaborative_VLM.git
cd Agent-Anchored_Collaborative_VLM
```

### 2) 创建虚拟环境

```bash
conda create -n aac-vlm python=3.10 -y
conda activate aac-vlm
```

### 3) 安装依赖

```bash
pip install torch==2.4.0 torch_npu== 2.4.0.post2 torchvision== 0.19.0
pip install transformers ms-swift deepspeed
pip install -r requirements.txt
```

## 📂 数据准备

请按照以下结构组织您的数据集。公开基准数据集（FFA-IR 和 MM-Retinal v1）可从各自的官方渠道获取。内部临床数据因伦理限制不公开。

```text
data/
├── ffa_ir/
│ ├── images/
│ └── annotations.jsonl
├── mm_retinal_v1/
└── custom_data/
├── images/
└── labels.jsonl
```

FFA-IR 的下载地址为：

MM-Retinal v1的下载地址为

## 📦 模型权重

## 🚀 快速开始

数据集为多图单报告，首先为每一张图像复制一份报告，整理为一对一的jsonl数据

Agent 辅助的句子级过滤

使用 EAA 提取图像的临床标签。

数据集Jsonl文件示例：

Example.jsonl:

```json
{"messages": [{"role": "user", "content": "你的身份是一名精通眼底疾病的眼科医生，你仔细阅读图像并书写文本描述。", "image": "/data/FFApic2/001583660002.png"}, {"role": "assistant", "content": "黄斑区血管迂曲，视盘上方可见WISS环。"}]}
```

修改文件路径并运行代码：

```bash
python fortagdata.py
```

模型会为所有数据集中所有图像附加上标签

运行后结果jsonl示例：

Example_taged.jsonl:

```json
{"messages": [{"role": "user", "content": "你的身份是一名精通眼底疾病的眼科医生，你仔细阅读图像并书写文本描述。", "image": "/data/FFApic2/001583660002.png"}, {"role": "assistant", "content": "黄斑区血管迂曲，视盘上方可见WISS环。"}], "tag": ["黄斑区", "中心凹", "视盘", "动脉", "静脉", "毛细血管", "后极部", "拱环"]}
```

修改路径并运行代码

Python selecttag.py

得到文本筛选后的数据集 filtered_Example.jsonl

VLM 报告生成

使用ms-swift的推理框架，整理数据集，并将prompt替换为第一阶段prompt

第一阶段prompt示例：

```json
{"messages": [{"role": "user", "content": "你的身份是一名精通眼底疾病的眼科医生，你拿到了一张右眼的，图像类型为FA的眼部图像的眼部图像，该患者性别为女，年龄为51岁，Series Time:142325，Acquisition Time:142446，经过模型初步分类，图像中可能包含以下标签：“黄斑区", "中心凹", "视盘", "动脉", "静脉", "毛细血管", "后极部", "拱环”，你仔细阅读图像并书写读片报告。", "image": "/data/FFApice2/001583660002.png"}, {"role": "assistant", "content": "黄斑区血管迂曲，视盘上方可见WISS环。"}]}
```

运行代码：

```bash
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=802816 \
swift infer \
--adapters /data/output/checkpoint-2161 \
--val_dataset /root/ filtered_Example_prompt1.jsonl \
--infer_backend pt \
--max_batch_size 1 \
--device_map auto \
--temperature 0.05 \
--repetition_penalty 1 \
--top_p 0.95 \
```

--max_new_tokens 512

第一阶段结果 OutputExample，重新组织训练数据集，将prompt替换为第二阶段prompt

第二阶段prompt示例：

```json
{"messages": [{"role": "user", "content": "你的身份是一名精通眼底疾病的眼科医生，你拿到了一张右眼的，图像类型为FA的眼部图像的眼部图像，该患者性别为女，年龄为51岁，Series Time:142325，Acquisition Time:142446，经过模型初步分类，图像中可能包含以下标签：“黄斑区", "中心凹", "视盘", "动脉", "静脉", "毛细血管", "后极部", "拱环”，你仔细阅读图像并书写读片报告并将书写的报告转换为标准化文本描述。规格化要求对原始描述进行分类整理，未出现在规格化规则内的内容保留，并且仅输出符合要求的文本描述，不增加其他解释或内容。规格化格式如下\n\n\n强荧光表现：必须选择以下具体标准描述中的一个或多个或保留原始表述。如强荧光表现是其他解剖结构的强荧光表现，需要告知是哪个解剖结构的强荧光表现：• “透见荧光”• “荧光着染”• “荧光积存”• “荧光渗漏”• “微血管瘤样高荧光”• “新生血管NV（+）”• “视盘新生血管NVD”• “斑片状高荧光”• “弥漫样高荧光”• “雪花样高荧光”• “组织着染”• “点簇样高荧光”• “下水道样高荧光”• “圆形高荧光”• “色素上皮改变样高荧光”• 其他表述；\n弱荧光表现：必须选择以下具体标准描述中的一个或多个或保留原始表述。如弱荧光表现是其他解剖结构的弱荧光表现，需要告知是哪个解剖结构的弱荧光表现：• “斑片状遮蔽荧光”• “散在点状遮蔽荧光”• “火焰状遮蔽荧光”• “荧光遮蔽”• “充盈障碍”• “低荧光”• “无灌注区NP”• 其他表述；\n视盘：必须选择以下具体标准描述中的一个或多个或保留原始表述： • “荧光充盈逐渐消退，无明显异常荧光渗漏” • “边界不清” • “色淡” • “C/D>0.3” • “视盘上方边缘欠清” • “呈高荧光” • “（半侧）荧光渗漏” • “特定侧（XX侧）荧光偏暗或不均” • 其他表述；\n黄斑区：必须选择以下具体标准描述中的一个或多个或保留原始表述： • “囊样荧光积存” • “拱环内囊样荧光积存” • “荧光渗漏” • “无明显血管，通常无荧光” • “可见毛细血管扩张” • 其他表述；\n血管：必须选择以下具体标准描述中的一个或多个或保留原始表述。如该血管是其他解剖结构的血管，需要告知是哪个解剖结构的血管： • “清晰可辨，无渗漏” • “清晰可辨，有渗漏” • “清晰可辨，有着染” • “清晰可辨，有闭塞” • “动脉充盈迟缓”或“动脉充盈延迟” • “静脉迂曲扩张，充盈迟缓” • “血管迂曲” • “血管纤细” • “周边血管闭塞，未见荧光充盈” • “异常血管袢” • “视网膜内微血管异常IRMA（+）” • “表面毛细血管扩张” • “静脉壁染” • “周边血管密集呈毛刷样” • “睫状动脉充盈迟缓” • 其他表述；\n其他解剖结构：必须选择以下具体标准描述中的一个或多个或保留原始表述： • “视网膜牵拉皱褶” • “网膜漂浮” • “周边可见变性区” • “虹膜表面呈高荧光” • 其他表述；\n屈光介质：必须选择以下具体标准描述中的一个或多个或保留原始表述： • “屈光间质欠清（晶体混）” • “玻腔影动（提示玻璃体混浊、出血、积血）” • “眼底朦胧” • “屈光间质不清” • “窥不入” • “玻璃体星状变性” • 其他表述；\n激光斑：必须选择以下具体标准描述中的一个或多个或保留原始表述： • “PRP清晰” • “NP区稍稀疏” • 其他表述；\n【其他病情描述:完全无法按照上述标准规格化的内容保留原始说明】\n• 其他表述，图像可能包含的信息如下：OutputExample ", "image": "/data/FFApice2/001583660002.png"}, {"role": "assistant", "content": "黄斑区血管迂曲，视盘上方可见WISS环。"}]}
```

运行代码：

```bash
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=802816 \
swift infer \
--adapters /data/output/checkpoint-2161 \
--val_dataset /root/ filtered_Example_prompt2.jsonl \
--infer_backend pt \
--max_batch_size 1 \
--device_map auto \
--temperature 0.05 \
--repetition_penalty 1 \
--top_p 0.95 \
```

--max_new_tokens 512

## 📝 完整流程

环境准备

克隆仓库

```bash
git clone https://github.com/SeuZL/Agent-Anchored_Collaborative_VLM.git
cd Agent-Anchored_Collaborative_VLM
```

创建虚拟环境

```bash
conda create -n aac-vlm python=3.10 -y
conda activate aac-vlm
```

安装依赖 (请补充具体的版本号)

```bash
pip install torch==<TODO: 版本> torchvision==<TODO: 版本>
pip install transformers ms-swift deepspeed
pip install -r requirements.txt
```

训练EAA

准备数据集，数据集组织方式如multi_label_data.jsonl所示

Python trainclassify.py

筛选模型训练数据

获取最优模型后

Python fortagdata.py

图像获取tag后

Python selecttag.py

获取到筛选后的数据集

训练基底模型FM

```bash
nohup bash -c 'export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_CHECK_TIMEOUT=600 MAX_PIXELS=802816 TORCH_NPU_FUSION_ENABLE=1; torchrun --nproc_per_node=8 --rdzv_conf "overlap_timeout=600" /root/zhangliang/Qwen2.5-VL/Lora/ms-swift/swift/cli/sft.py --ddp_backend hccl --model /root/Qwen2.5vl-32B-Instruct --dataset /root/Qwen2.5-VL/yanke/ori_selected_withtag_newprompt.jsonl --deepspeed /root/ /Qwen2.5-VL/yanke/deepspeedzero3.json --train_type lora --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.1 --weight_decay 0.01 --lora_rank 32 --lora_alpha 64 --lora_dropout 0.1 --use_rslora False --max_grad_norm 1.0 --target_modules all-linear --freeze_vit true --eval_steps 100 --save_steps 100 --save_total_limit 100 --logging_steps 50 --max_length 8192 --output_dir /data/output/yanke --dataloader_num_workers 8 --model_kwargs "{\"device_map\":{\"\": \"npu:auto\"}}" --optim adamw_hf --gradient_checkpointing True' > /data/logs/qwen2.5lora/alldata32B.txt 2>&1 &
```

获取最优模型

使用模型进行推理

可选取ms-swift的推理demo

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
--adapters /data /checkpoint-2161 \
--merge_lora true \
```

--device_map auto

也可以对组织好的数据集进行推理

数据集组织方式可参考infer_example.jsonl

```bash
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=802816 \
swift infer \
--adapters /data/output/checkpoint-2161 \
--val_dataset /root/ filtered_Example.jsonl \
--infer_backend pt \
--max_batch_size 1 \
--device_map auto \
--temperature 0.05 \
--repetition_penalty 1 \
--top_p 0.95 \
```

--max_new_tokens 512

对模型进行评测

我们使用了ms-swift的推理框架

评测数据集的组织可参考eval_example.jsonl

```bash
NPROC_PER_NODE=8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=802816 \
swift eval \
--model /data/checkpoint-2161-merged \
--eval_backend Native \
--infer_backend pt \
--device_map auto \
--eval_limit 300 \
--eval_dataset general_qa \
```

--dataset_args '{"general_qa": {"local_path": "/root/Qwen2.5-VL/yanke/qa", "subset_list": ["eval_example"]}}'

## 🤝 致谢

本研究获得“东南大学—江苏省人民医院联合开放器官芯片项目”（2024-K01）以及“江苏省人民医院专病临床研究基金”（XB202404）资助。
