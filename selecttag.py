import json
import os
from tqdm import tqdm

# 配置
input_file = '/root/zhangliang/Qwen2.5-VL/yanke/infer_tagged_data.jsonl'  # 你的输入文件
output_file = '/root/zhangliang/Qwen2.5-VL/yanke/filtered_data.jsonl'      # 输出文件

# 标签表
L = [
    "窥不入", "影动", "黄斑区", "中心凹", "视盘", "动脉", "静脉", "毛细血管", "网膜下遮蔽", "后极部",
    "中周部", "眼周边", "拱环", "高荧光", "低荧光", "荧光着染", "遮蔽荧光", "积存", "色素上皮",
    "血管壁染", "荧光渗漏", "透见荧光", "点状", "片状", "簇样", "点片状", "微血管瘤样", "不规则",
    "弥漫样", "息肉样", "变性区", "异常血管", "萎缩灶", "激光斑", "出血", "呈高度近视眼底改变",
    "网膜漂浮", "NP", "NV", "NVD"
]

# 开始处理
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    
    lines = fin.readlines()
    for line in tqdm(lines, desc="筛选处理中"):
        message = json.loads(line.strip())
        tags_in_message = message.get('tag', [])
        
        # 找到 assistant 的 content
        assistant_contents = []
        for msg in message['messages']:
            if msg['role'] == 'assistant':
                assistant_contents.append(msg['content'])
        
        if not assistant_contents:
            continue  # 如果没有assistant内容直接跳过
        
        new_assistant_contents = []
        
        for content in assistant_contents:
            # 按逗号和句号分割
            small_sentences = []
            for fragment in content.split('。'):
                small_sentences.extend(fragment.split('，'))
            small_sentences = [s.strip() for s in small_sentences if s.strip()]

            # 逐个小句处理
            kept_sentences = []
            for sentence in small_sentences:
                has_L_tag = any(tag in sentence for tag in L)
                if not has_L_tag:
                    kept_sentences.append(sentence)
                else:
                    has_B_tag = any(tag in sentence for tag in tags_in_message)
                    if has_B_tag:
                        kept_sentences.append(sentence)

            if kept_sentences:
                # 将保留的小句重新拼成content
                new_content = '，'.join(kept_sentences) + '。'
                new_assistant_contents.append(new_content)

        if not new_assistant_contents:
            continue  # 如果处理完后assistant没有内容，则整条message不要
        
        # 替换assistant内容
        new_messages = []
        for msg in message['messages']:
            if msg['role'] == 'assistant':
                msg['content'] = new_assistant_contents.pop(0)
            new_messages.append(msg)
        
        message['messages'] = new_messages

        fout.write(json.dumps(message, ensure_ascii=False) + '\n')

print(f"✅ 筛选完成，输出到 {output_file}")
