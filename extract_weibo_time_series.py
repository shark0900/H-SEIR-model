#!/usr/bin/env python3
"""
从CED_Dataset提取真实的时间序列数据
用于H-SEIR模型拟合

数据处理流程:
1. 读取original-microblog中的谣言帖子 (获取发布时间)
2. 读取rumor-repost中的转发数据 (获取转发时间)
3. 构建I(t)时间序列 (每小时转发数)
4. 保存为CSV格式供模型拟合
"""

import json
import os
from datetime import datetime
from collections import defaultdict
import csv

def parse_timestamp(ts):
    """解析时间戳，支持多种格式"""
    if isinstance(ts, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(ts)
    elif isinstance(ts, str):
        # 字符串格式: "2012-09-16 15:05:11"
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except:
            return None
    return None

def extract_time_series(rumor_id, orig_file, repost_file):
    """
    为单个谣言提取时间序列
    
    返回:
        rumor_id: 谣言ID
        orig_time: 原始帖子发布时间
        repost_times: 转发时间列表
        I_t: 每小时的转发数 (时间序列)
    """
    # 读取原始帖子
    orig_data = None
    if os.path.exists(orig_file):
        with open(orig_file, 'r', encoding='utf-8') as f:
            orig_data = json.load(f)
    
    if not orig_data:
        return None
    
    # 获取原始帖子发布时间
    orig_time = None
    if 'time' in orig_data:
        orig_time = parse_timestamp(orig_data['time'])
    
    if not orig_time:
        return None
    
    # 读取转发数据
    repost_times = []
    if os.path.exists(repost_file):
        with open(repost_file, 'r', encoding='utf-8') as f:
            reposts = json.load(f)
        
        for repost in reposts:
            if 'date' in repost:
                rt = parse_timestamp(repost['date'])
                if rt:
                    repost_times.append(rt)
    
    if not repost_times:
        return None
    
    # 构建时间序列 (每小时转发数)
    time_series = defaultdict(int)
    for rt in repost_times:
        # 计算距离原始帖子的小时数
        hours_diff = int((rt - orig_time).total_seconds() / 3600)
        if hours_diff >= 0:  # 只算原始帖子发布后的转发
            time_series[hours_diff] += 1
    
    return {
        'rumor_id': rumor_id,
        'orig_time': orig_time,
        'total_reposts': len(repost_times),
        'time_series': dict(time_series)
    }

def main():
    base_dir = r"D:\论文\H-SEIR输出\Chinese_Rumor_Dataset\CED_Dataset"
    orig_dir = os.path.join(base_dir, "original-microblog")
    repost_dir = os.path.join(base_dir, "rumor-repost")
    
    output_dir = r"D:\论文\H-SEIR输出\real_weibo_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 开始提取真实Weibo时间序列数据 ===\n")
    
    # 获取所有谣言文件
    rumor_files = [f for f in os.listdir(repost_dir) if f.endswith('.json')]
    print(f"找到 {len(rumor_files)} 个谣言的转发数据\n")
    
    all_time_series = []
    valid_count = 0
    
    for i, fname in enumerate(rumor_files[:100]):  # 先处理前100个
        rumor_id = fname.replace('.json', '')
        orig_file = os.path.join(orig_dir, fname)
        repost_file = os.path.join(repost_dir, fname)
        
        result = extract_time_series(rumor_id, orig_file, repost_file)
        
        if result and result['total_reposts'] > 10:  # 至少10条转发
            all_time_series.append(result)
            valid_count += 1
            
            if valid_count <= 3:  # 打印前3个示例
                print(f"示例 {valid_count}:")
                print(f"  ID: {result['rumor_id']}")
                print(f"  发布时间: {result['orig_time']}")
                print(f"  总转发数: {result['total_reposts']}")
                print(f"  时间序列长度: {len(result['time_series'])} 小时")
                print()
    
    print(f"=== 提取完成 ===")
    print(f"有效谣言数: {valid_count}")
    
    # 保存为CSV
    csv_file = os.path.join(output_dir, "weibo_time_series.csv")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rumor_id', 'orig_time', 'hour', 'reposts'])
        
        for item in all_time_series:
            for hour, count in sorted(item['time_series'].items()):
                writer.writerow([item['rumor_id'], item['orig_time'], hour, count])
    
    print(f"\n数据已保存到: {csv_file}")
    print(f"总记录数: {sum(len(item['time_series']) for item in all_time_series)}")
    
    # 保存汇总统计
    stats_file = os.path.join(output_dir, "weibo_stats.json")
    stats = {
        'total_rumors': valid_count,
        'total_reposts': sum(item['total_reposts'] for item in all_time_series),
        'avg_reposts_per_rumor': sum(item['total_reposts'] for item in all_time_series) / valid_count if valid_count > 0 else 0
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"统计信息已保存到: {stats_file}")
    
    return all_time_series

if __name__ == "__main__":
    main()
