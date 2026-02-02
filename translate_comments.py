#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch translate Chinese comments to English in Python files
"""

import re
import os

# Translation dictionary for common Chinese comments
translations = {
    # Common phrases
    "配置": "Configuration",
    "参数": "Parameters",
    "路径": "Path",
    "模型": "Model",
    "数据集": "Dataset",
    "加载": "Load",
    "保存": "Save",
    "输出": "Output",
    "评测": "Evaluation",
    "评估": "Evaluation",
    "训练": "Training",
    "推理": "Inference",
    "生成": "Generation",
    "结果": "Results",
    "统计": "Statistics",
    "计算": "Calculate",
    "函数": "Function",
    "主函数": "Main Function",
    "核心": "Core",
    "逻辑": "Logic",
    "步骤": "Step",
    "开始": "Start",
    "完成": "Complete",
    "成功": "Success",
    "失败": "Failed",
    "错误": "Error",
    "警告": "Warning",
    "注意": "Note",
    "示例": "Example",
    "循环": "Loop",
    "遍历": "Iterate",
    "处理": "Process",
    "合并": "Merge",
    "分片": "Shard",
    "权重": "Weight",
    "基础模型": "Base model",
    "适配器": "Adapter",
    "目录": "Directory",
    "文件": "File",
    "创建": "Create",
    "复制": "Copy",
    "检查": "Check",
    "确认": "Confirm",
    "格式": "Format",
    "索引": "Index",
    "组织": "Organize",
    "查找": "Lookup",
    "匹配": "Match",
    "替换": "Replace",
    "转换": "Convert",
    "归一化": "Normalize",
    "分数": "Score",
    "概率": "Probability",
    "准确率": "Accuracy",
    "最终": "Final",
    "总": "Total",
    "有效": "Valid",
    "数量": "Count",
    "胜率": "Win rate",
    "平均": "Average",
    "长度": "Length",
    "清理": "Clean",
    "后": "After",
    "基于": "Based on",
    "回复": "Response",
    "查询": "Query",
    "调用": "Call",
    "准备": "Prepare",
    "进行": "Perform",
    "获得": "Obtain",
    "没有": "No",
    "任何": "Any",
    "所有": "All",
    "每个": "Each",
    "对": "For",
    "的": "",
    "和": "and",
    "或者": "or",
    "如果": "If",
    "否则": "Otherwise",
    "因为": "Because",
    "但": "But",
    "这里": "Here",
    "这是": "This is",
    "用于": "Used for",
    "需要": "Need",
    "可以": "Can",
    "必须": "Must",
    "应该": "Should",
    "将": "Will",
    "已": "Already",
    "正在": "Currently",
    "请": "Please",
    "修改为": "Modify to",
    "替换成": "Replace with",
    "你的": "Your",
    "我们": "We",
    "它": "It",
    "是": "Is",
    "在": "In",
    "到": "To",
    "从": "From",
    "以": "With",
    "为了": "For",
    "通过": "Through",
    "使用": "Use",
    "设置": "Set",
    "获取": "Get",
    "返回": "Return",
    "包含": "Contains",
    "存在": "Exists",
    "找不到": "Cannot find",
    "找到了": "Found",
    "个": "",
    "等于": "Equals",
    "大于": "Greater than",
    "小于": "Less than",
    "不": "Not",
    "非": "Non",
    "自动": "Auto",
    "手动": "Manual",
    "简单": "Simple",
    "复杂": "Complex",
    "标准": "Standard",
    "自定义": "Custom",
    "特殊": "Special",
    "通常": "Usually",
    "可能": "May",
    "假设": "Assume",
    "依据": "Based on",
    "提供": "Provide",
    "代码": "Code",
    "引入": "Import",
    "包": "Package",
    "当前": "Current",
    "路径": "Path",
    "中": "In",
    "确保": "Ensure",
    "文件夹": "Folder",
    "位置": "Position",
    "预测": "Prediction",
    "标签": "Label",
    "真实": "Actual",
    "选项": "Choice",
    "问题": "Question",
    "答案": "Answer",
    "正确": "Correct",
    "错误": "Wrong",
    "总概率": "Total probability",
    "定义": "Definition",
    "实际上": "Actually",
    "做法": "Practice",
    "理解": "Understanding",
    "实现": "Implementation",
    "分布": "Distribution",
    "求": "Calculate",
    "之和": "Sum",
    "写入": "Write",
    "最终统计": "Final statistics",
    "简单起见": "For simplicity",
    "快": "Fast",
    "复用": "Reuse",
    "模板": "Template",
    "通用性": "Generality",
    "拼接": "Concatenation",
    "标记": "Marker",
    "编码": "Encode",
    "构造": "Construct",
    "输入": "Input",
    "额外": "Additional",
    "根据": "Based on",
    "签名": "Signature",
    "关心": "Care about",
    "部分": "Part",
    "即": "I.e.",
    "之后": "After",
    "对应": "Correspond",
    "当前位置": "Current position",
    "防止": "Prevent",
    "长选项": "Long choices",
    "得分": "Score",
    "过低": "Too low",
    "连乘": "Multiplication",
    "单进程": "Single process",
    "分布式": "Distributed",
    "打分": "Score",
    "转换": "Convert",
    "假设": "Assume",
    "归一化": "Normalized",
    "理解": "Understanding",
    "一种": "One",
    "总": "Total",
    "集合": "Set",
    "赋予": "Assigned",
}

def translate_comment(text):
    """Simple translation of Chinese comments"""
    # Skip if no Chinese characters
    if not re.search(r'[\u4e00-\u9fa5]', text):
        return text
    
    # Try to translate common patterns
    result = text
    for cn, en in translations.items():
        if cn in result:
            result = result.replace(cn, en)
    
    return result

def process_file(filepath):
    """Process a single Python file"""
    print(f"Processing: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        
        for line in lines:
            # Check if line contains Chinese
            if re.search(r'[\u4e00-\u9fa5]', line):
                # Translate the line
                new_line = translate_comment(line)
                new_lines.append(new_line)
                modified = True
            else:
                new_lines.append(line)
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"  ✓ Updated")
        else:
            print(f"  - No Chinese found")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

def main():
    base_dir = "/Users/linlinliang/Desktop/AR-MAP"
    
    # Files to process
    files = [
        "eval-dream/dream-truthful.py",
        "eval-dream/dream-helpful.py",
        "eval-qwen/help_eval.py",
        "eval-qwen/arena_qwen3.py",
        "eval-sdar/help_eval_sdar.py",
        "eval-sdar/arena_sdar.py",
        "eval-sdar/sdar_truthful.py",
        "eval-sdar/ifeval_eval_sdar.py",
        "merge-lora-dream.py",
        "merge-lora-sdar.py",
    ]
    
    for file in files:
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()
