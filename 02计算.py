import pandas as pd

def analyze_simulation_results(file_path='full_data_analysis.csv'):
    # 1. 加载数据
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载文件: {file_path}")
    except FileNotFoundError:
        print("未找到结果文件，请先运行模拟脚本生成 CSV。")
        return

    # ==========================================
    # 2. 规则影响统计 (Impact Statistics)
    # ==========================================
    # 统计总周次 (去重 Season + Week)
    total_weeks = df[['Season', 'Week']].drop_duplicates().shape[0]
    
    # 统计结果改变的周次 (Outcome_Changed == True)
    changed_weeks_df = df[df['Outcome_Changed'] == True][['Season', 'Week']].drop_duplicates()
    num_changed_weeks = changed_weeks_df.shape[0]
    
    impact_rate = (num_changed_weeks / total_weeks) * 100
    
    print(f"\n=== 1. 规则改变影响统计 ===")
    print(f"总分析周次: {total_weeks}")
    print(f"结果改变周次: {num_changed_weeks}")
    print(f"规则影响率: {impact_rate:.2f}%")
    
    # ==========================================
    # 3. 受害者与受益者筛选 (Victims & Beneficiaries)
    # ==========================================
    # 受害者: 在实际规则下被淘汰，但在反事实规则下本可幸存
    victims = df[
        (df['Result_Actual'] == 'Eliminated') & 
        (df['Result_Counter'] == 'Safe')
    ]
    
    # 受益者: 在实际规则下幸存，但在反事实规则下本会被淘汰
    beneficiaries = df[
        (df['Result_Actual'] == 'Safe') & 
        (df['Result_Counter'] == 'Eliminated')
    ]
    
    print(f"\n=== 2. 具体受影响人数 ===")
    print(f"规则受害者 (Victims): {len(victims)} 人")
    print(f"规则受益者 (Beneficiaries): {len(beneficiaries)} 人")
    
    # 保存名单以便查看
    victims.to_csv('list_of_victims.csv', index=False)
    beneficiaries.to_csv('list_of_beneficiaries.csv', index=False)
    print("(详细名单已保存至 list_of_victims.csv 和 list_of_beneficiaries.csv)")

    # ==========================================
    # 4. 偏差分析 (Bias Analysis)
    # ==========================================
    # 目的: 在结果不同的周次中，哪种规则“杀死了”粉丝票最低的人？
    
    pct_favors_fans = 0
    rank_favors_fans = 0
    
    # 遍历每一个发生改变的周次
    changed_weeks_list = changed_weeks_df.values.tolist()
    
    for season, week in changed_weeks_list:
        # 获取该周数据
        week_data = df[(df['Season'] == season) & (df['Week'] == week)]
        
        # 获取该周使用的规则名称
        actual_method = week_data['Actual_Method'].iloc[0]
        counter_method = week_data['Counter_Method'].iloc[0]
        
        # 检查 Bias 标记
        # Bias_Actual_Kill_FanLoser = True 表示实际规则淘汰了当周人气最低者
        # Bias_Counter_Kill_FanLoser = True 表示反事实规则淘汰了当周人气最低者
        
        if week_data['Bias_Actual_Kill_FanLoser'].any():
            if actual_method == 'percentage':
                pct_favors_fans += 1
            elif actual_method == 'rank':
                rank_favors_fans += 1
                
        if week_data['Bias_Counter_Kill_FanLoser'].any():
            if counter_method == 'percentage':
                pct_favors_fans += 1
            elif counter_method == 'rank':
                rank_favors_fans += 1
                
    print(f"\n=== 3. 偏差分析 (Bias Analysis) ===")
    print(f"在 {num_changed_weeks} 个结果不同的周次中:")
    print(f"百分比法 (Percentage) 顺从粉丝意愿 (淘汰人气最低者): {pct_favors_fans} 次")
    print(f"排名法 (Rank) 顺从粉丝意愿 (淘汰人气最低者):       {rank_favors_fans} 次")
    
    if pct_favors_fans > rank_favors_fans:
        print("-> 结论: 百分比法赋予粉丝投票的权重更大 (Favors Popular Vote)。")
    else:
        print("-> 结论: 排名法赋予粉丝投票的权重更大。")

# 运行分析
if __name__ == "__main__":
    analyze_simulation_results()