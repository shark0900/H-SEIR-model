#!/usr/bin/env python3
"""
用真实的Weibo时间序列数据拟合H-SEIR模型
获得真实的R²值（不是编造的）

流程:
1. 读取真实的时间序列数据 (weibo_time_series.csv)
2. 对每个谣言拟合H-SEIR ODE系统
3. 计算R² (真实值)
4. 生成拟合图 (Fig_Weibo_Fit.png)
5. 输出统计结果
"""

import csv
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def h_seir_ode(t, y, beta0, gamma, alpha, theta, tau, epsilon):
    """
    H-SEIR ODE系统
    
    参数:
        beta0: 基础传播率
        gamma: 恢复率
        alpha: 心理调节参数
        theta: 认知反思阈值 (均值)
        tau: 信任倾向 (均值)
        epsilon: 情绪易感性 (均值)
    """
    S, E, I, R = y
    N = S + E + I + R
    
    # 心理调节的传播率
    beta_eff = beta0 * (1 + alpha * tau) * (1 + alpha * epsilon) / (1 + alpha * theta)
    
    dSdt = -beta_eff * S * I / N
    dEdt = beta_eff * S * I / N - gamma * E
    dIdt = gamma * E - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dEdt, dIdt, dRdt]

def fit_h_seir(time_series, rumor_id):
    """
    对单个谣言的时间序列拟合H-SEIR模型
    
    返回:
        R2: 拟合优度
        params: 拟合参数
        I_pred: 预测的感染曲线
    """
    # 解析时间序列
    hours = sorted([int(k) for k in time_series.keys()])
    I_obs = [time_series[str(h)] if str(h) in time_series else time_series[h] for h in hours]
    
    # 归一化 (转化为感染比例)
    total_reposts = sum(I_obs)
    I_norm = np.array(I_obs) / max(total_reposts, 1)
    
    # 初始条件
    N = 10000  # 假设的总人口
    I0 = I_norm[0] if len(I_norm) > 0 else 0.001
    S0 = 1 - I0
    E0 = 0
    R0 = 0
    y0 = [S0, E0, I0, R0]
    
    # 时间网格
    t_span = (0, max(hours))
    t_eval = np.linspace(0, max(hours), len(hours))
    
    # 初始参数猜测
    beta0_init = 0.5
    gamma_init = 0.1
    alpha_init = 0.3
    theta_init = 0.4
    tau_init = 0.5
    epsilon_init = 0.6
    
    # 定义目标函数
    def objective(t, beta0, gamma, alpha, theta, tau, epsilon):
        sol = solve_ivp(
            h_seir_ode, t_span, y0,
            args=(beta0, gamma, alpha, theta, tau, epsilon),
            t_eval=t_eval,
            method='RK45'
        )
        return sol.y[2]  # 返回I(t)
    
    try:
        # 曲线拟合
        popt, pcov = curve_fit(
            objective, hours, I_norm,
            p0=[beta0_init, gamma_init, alpha_init, theta_init, tau_init, epsilon_init],
            bounds=([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]),
            maxfev=5000
        )
        
        # 预测
        I_pred = objective(t_eval, *popt)
        
        # 计算R²
        SS_res = np.sum((I_norm - I_pred) ** 2)
        SS_tot = np.sum((I_norm - np.mean(I_norm)) ** 2)
        R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0
        
        params = {
            'beta0': popt[0],
            'gamma': popt[1],
            'alpha': popt[2],
            'theta': popt[3],
            'tau': popt[4],
            'epsilon': popt[5]
        }
        
        return R2, params, I_pred, I_norm, t_eval
        
    except Exception as e:
        print(f"  拟合失败: {e}")
        return None, None, None, None, None

def main():
    # 读取真实时间序列数据
    data_file = r"D:\论文\H-SEIR输出\real_weibo_data\weibo_time_series.csv"
    output_dir = r"D:\论文\H-SEIR输出\real_weibo_fit"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 用真实Weibo数据拟合H-SEIR模型 ===\n")
    
    # 按rumor_id聚合时间序列
    rumors_data = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row['rumor_id']
            hour = int(row['hour'])
            reposts = int(row['reposts'])
            
            if rid not in rumors_data:
                rumors_data[rid] = {}
            rumors_data[rid][hour] = reposts
    
    print(f"加载了 {len(rumors_data)} 个谣言的时间序列\n")
    
    # 拟合前10个谣言 (作为示例)
    results = []
    R2_values = []
    
    for i, (rid, time_series) in enumerate(list(rumors_data.items())[:10]):
        print(f"[{i+1}/10] 拟合谣言: {rid}")
        print(f"  时间序列长度: {len(time_series)} 小时")
        
        R2, params, I_pred, I_obs, t_eval = fit_h_seir(time_series, rid)
        
        if R2 is not None:
            print(f"  拟合成功! R² = {R2:.4f}")
            R2_values.append(R2)
            results.append({
                'rumor_id': rid,
                'R2': R2,
                'params': params
            })
        else:
            print(f"  拟合失败")
        print()
    
    # 统计结果
    if R2_values:
        avg_R2 = np.mean(R2_values)
        std_R2 = np.std(R2_values)
        min_R2 = np.min(R2_values)
        max_R2 = np.max(R2_values)
        
        print("=== 拟合结果统计 ===")
        print(f"成功拟合: {len(R2_values)} 个谣言")
        print(f"平均 R²: {avg_R2:.4f}")
        print(f"标准差 R²: {std_R2:.4f}")
        print(f"R² 范围: [{min_R2:.4f}, {max_R2:.4f}]")
        print()
        
        # 保存结果
        result_file = os.path.join(output_dir, "weibo_fit_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"拟合结果已保存到: {result_file}")
        
        # 生成拟合图 (第一个谣言)
        if len(results) > 0:
            rid = results[0]['rumor_id']
            time_series = rumors_data[rid]
            
            hours = sorted(time_series.keys())
            I_obs = [time_series[h] for h in hours]
            
            R2, params, I_pred, I_obs_norm, t_eval = fit_h_seir(time_series, rid)
            
            plt.figure(figsize=(10, 6))
            plt.plot(hours, I_obs, 'bo-', label='Observed (真实数据)', linewidth=2, markersize=6)
            plt.plot(t_eval, I_pred * max(I_obs), 'r--', label=f'H-SEIR Fit (R²={R2:.4f})', linewidth=2)
            plt.xlabel('Time (hours)', fontsize=12)
            plt.ylabel('Number of Reposts', fontsize=12)
            plt.title(f'H-SEIR Fit to Real Weibo Data\nRumor ID: {rid[:20]}...', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            fig_file = os.path.join(output_dir, "Fig_Weibo_Fit.png")
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n拟合图已保存到: {fig_file}")
        
        return avg_R2, std_R2
    else:
        print("错误: 所有拟合都失败了!")
        return None, None

if __name__ == "__main__":
    avg_R2, std_R2 = main()
