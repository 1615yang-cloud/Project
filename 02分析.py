import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

# ==========================================
# 0. 配置：争议人物列表 (The Controversial 10)
# ==========================================
# 格式: (Name, Season, Description)
CONTROVERSIAL_FIGURES = [
    # --- 组别 A: 低分高人气 (靠粉丝续命的“不死鸟”) ---
    ("Jerry Rice", 2, "Ranking Era Survivor"),
    ("Bobby Bones", 27, "Percentage Era Winner (The Catalyst for Rule Change)"),
    ("Bristol Palin", 11, "Percentage Era Survivor"),
    ("Sean Spicer", 28, "Ranking Era Survivor"),
    ("Bill Engvall", 17, "Percentage Era Finalist (Low Scores)"),
    ("Marie Osmond", 5, "Percentage Era Finalist (Huge Fanbase)"),
    
    # --- 组别 B: 高分低人气 (规则/粉丝投票的“受害者”) ---
    ("Sabrina Bryan", 5, "The Original Shock Elimination"),
    ("Willow Shields", 20, "Shock Elimination"),
    ("Juan Pablo Di Pace", 27, "Perfect Score Elimination"),
    ("Heather Morris", 24, "Professional Dancer Shock Elimination")
]

# ==========================================
# 1. 核心计算模型 (Scoring & Constraints)
# ==========================================

def calculate_scores(judges_scores, fan_votes, method):
    """根据指定方法计算总分"""
    if method == 'percentage':
        # 百分比法
        if np.sum(judges_scores) == 0:
            judge_share = np.zeros_like(judges_scores)
        else:
            judge_share = judges_scores / np.sum(judges_scores)
        return judge_share + fan_votes
        
    elif method == 'rank':
        # 排名积分法
        judge_points = stats.rankdata(judges_scores, method='min')
        fan_points = stats.rankdata(fan_votes, method='min')
        return judge_points + fan_points

def check_constraints(total_scores, fan_votes, eliminated_indices, method):
    """检查是否满足历史淘汰约束"""
    if not eliminated_indices: return True
    n = len(total_scores)
    survivor_indices = [i for i in range(n) if i not in eliminated_indices]
    if not survivor_indices: return True
    
    s_scores = total_scores[survivor_indices]
    e_scores = total_scores[eliminated_indices]
    
    # 快速检查
    if np.min(s_scores) > np.max(e_scores): return True
    
    if method == 'percentage':
        return np.min(s_scores) >= np.max(e_scores)
    else:
        # 排名法同分决胜
        S_scores_grid = s_scores[:, np.newaxis]
        E_scores_grid = e_scores[np.newaxis, :]
        S_votes_grid = fan_votes[survivor_indices][:, np.newaxis]
        E_votes_grid = fan_votes[eliminated_indices][np.newaxis, :]
        fail = (S_scores_grid < E_scores_grid) | \
               ((S_scores_grid == E_scores_grid) & (S_votes_grid <= E_votes_grid))
        return not np.any(fail)

def get_simulated_losers(scores, fan_votes):
    """找出模拟环境下的淘汰者"""
    min_score = np.min(scores)
    candidates = np.where(scores == min_score)[0]
    if len(candidates) == 1: return candidates
    # Tie-breaker: Lowest Fan Vote
    candidate_votes = fan_votes[candidates]
    min_vote = np.min(candidate_votes)
    losers = candidates[np.where(candidate_votes == min_vote)[0]]
    return losers

# ==========================================
# 2. 全量分析主程序
# ==========================================

def run_full_analysis():
    print("正在加载全量数据...")
    try:
        df = pd.read_csv('cleaned_DWTS_data.csv')
    except FileNotFoundError:
        print("错误: 未找到 cleaned_DWTS_data.csv")
        return

    # 数据预处理
    df['eliminated_week'] = pd.to_numeric(
        df['eliminated_week'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )
    
    full_results = []
    
    # 扫描所有赛季
    seasons = sorted(df['season'].unique())
    print(f"将分析 {len(seasons)} 个赛季的所有数据...")
    
    # 模拟参数 (全量分析需要兼顾速度与准确性)
    n_samples = 600
    burn_in = 100
    
    for season in seasons:
        # 确定规则
        if 3 <= season <= 27:
            actual_method = 'percentage'
            counter_method = 'rank'
        else:
            actual_method = 'rank'
            counter_method = 'percentage'
            
        cols = [c for c in df.columns if c.startswith('total_score_w')]
        weeks = sorted([int(c.split('w')[1]) for c in cols])
        
        for week in weeks:
            score_col = f'total_score_w{week}'
            if score_col not in df.columns: continue
            
            current_df = df[df['season'] == season].copy()
            current_df = current_df[current_df[score_col] > 0].reset_index(drop=True)
            if current_df.empty: continue
            
            names = current_df['celebrity_name'].values
            judges_scores = current_df[score_col].values
            n_contestants = len(names)
            if n_contestants < 2: continue
            
            eliminated_mask = current_df['eliminated_week'] == week
            eliminated_indices = np.where(eliminated_mask)[0].tolist()
            if not eliminated_indices: continue
            
            # --- MCMC 推断 ---
            samples = []
            current_votes = np.random.dirichlet(np.ones(n_contestants))
            valid_start = False
            for _ in range(200):
                sc = calculate_scores(judges_scores, current_votes, actual_method)
                if check_constraints(sc, current_votes, eliminated_indices, actual_method):
                    valid_start = True
                    break
                current_votes = np.random.dirichlet(np.ones(n_contestants))
            
            if not valid_start: continue
            
            for i in range(n_samples + burn_in):
                prop = current_votes.copy()
                idx1, idx2 = np.random.choice(n_contestants, 2, replace=False)
                step = np.random.uniform(0, 0.05)
                if prop[idx1] > step:
                    prop[idx1] -= step; prop[idx2] += step
                else:
                    prop[idx1] += step; prop[idx2] -= step
                
                sc_prop = calculate_scores(judges_scores, prop, actual_method)
                if check_constraints(sc_prop, prop, eliminated_indices, actual_method):
                    current_votes = prop
                
                if i >= burn_in:
                    samples.append(current_votes.copy())
            
            if not samples: continue
            mean_votes = np.mean(samples, axis=0)
            
            # --- 反事实模拟 ---
            scores_actual = calculate_scores(judges_scores, mean_votes, actual_method)
            scores_counter = calculate_scores(judges_scores, mean_votes, counter_method)
            losers_actual = get_simulated_losers(scores_actual, mean_votes)
            losers_counter = get_simulated_losers(scores_counter, mean_votes)
            
            set_actual = set(losers_actual)
            set_counter = set(losers_counter)
            outcome_changed = (set_actual != set_counter)
            
            lowest_fan_idx = np.argmin(mean_votes)
            
            # 记录所有人
            for idx, name in enumerate(names):
                # 检查是否是争议人物 (模糊匹配)
                is_target = any(t in name for t,_,_ in CONTROVERSIAL_FIGURES)
                
                st_act = "Eliminated" if idx in losers_actual else "Safe"
                st_cnt = "Eliminated" if idx in losers_counter else "Safe"
                
                vote_rank = n_contestants - stats.rankdata(mean_votes, method='min')[idx] + 1
                judge_rank = n_contestants - stats.rankdata(judges_scores, method='min')[idx] + 1
                
                # 偏差判定
                bias_actual = False
                bias_counter = False
                if outcome_changed:
                    if idx == lowest_fan_idx and st_act == "Eliminated": bias_actual = True
                    if idx == lowest_fan_idx and st_cnt == "Eliminated": bias_counter = True

                full_results.append({
                    'Season': season,
                    'Week': week,
                    'Contestant': name,
                    'Is_Controversial': is_target,
                    'Actual_Method': actual_method,
                    'Counter_Method': counter_method,
                    'Judges_Score': judges_scores[idx],
                    'Judge_Rank': judge_rank,
                    'Inferred_Fan_Vote': mean_votes[idx],
                    'Fan_Vote_Rank': vote_rank,
                    'Result_Actual': st_act,
                    'Result_Counter': st_cnt,
                    'Outcome_Changed': outcome_changed,
                    'Bias_Actual_Kill_FanLoser': bias_actual,
                    'Bias_Counter_Kill_FanLoser': bias_counter
                })
                
    df_res = pd.DataFrame(full_results)
    df_res.to_csv('full_data_analysis.csv', index=False)
    
    # ==========================================
    # 3. 输出统计简报
    # ==========================================
    print("\n=== 全量数据分析简报 ===")
    total_weeks = df_res[['Season', 'Week']].drop_duplicates().shape[0]
    changed_weeks = df_res[df_res['Outcome_Changed']==True][['Season', 'Week']].drop_duplicates().shape[0]
    print(f"分析总周次: {total_weeks}")
    print(f"结果改变周次: {changed_weeks} ({changed_weeks/total_weeks:.1%})")
    
    # 偏差统计
    diff_rows = df_res[df_res['Outcome_Changed']==True]
    method_bias = {'percentage': 0, 'rank': 0}
    
    # 遍历改变的周次，看是谁杀了粉丝票最低者
    if not diff_rows.empty:
        changed_week_list = diff_rows[['Season', 'Week']].drop_duplicates().values
        for s, w in changed_week_list:
            week_data = diff_rows[(diff_rows['Season']==s) & (diff_rows['Week']==w)]
            act_meth = week_data['Actual_Method'].iloc[0]
            cnt_meth = week_data['Counter_Method'].iloc[0]
            
            if week_data['Bias_Actual_Kill_FanLoser'].any():
                method_bias[act_meth] += 1
            elif week_data['Bias_Counter_Kill_FanLoser'].any():
                method_bias[cnt_meth] += 1
            
    print(f"\n[偏差分析] 在 {changed_weeks} 个结果不同的周次中:")
    print(f"- 百分比法 (Percentage) 顺从粉丝意愿: {method_bias['percentage']} 次")
    print(f"- 排名法 (Rank) 顺从粉丝意愿:       {method_bias['rank']} 次")
    
    # ==========================================
    # 4. 争议人物深度扫描报告
    # ==========================================
    print("\n=== 争议人物深度扫描 (Top 10 Analysis) ===")
    
    controversy_summary = []
    
    # 为了准确匹配，我们遍历预定义的列表
    for t_name, t_season, _ in CONTROVERSIAL_FIGURES:
        # 在结果中找到该人 (模糊匹配名字且匹配赛季)
        # 注意: full_results 里存的是全名，这里做包含匹配
        sub = df_res[
            (df_res['Season'] == t_season) & 
            (df_res['Contestant'].str.contains(t_name, case=False))
        ]
        
        if sub.empty:
            print(f"> {t_name}: 未找到数据 (可能是拼写差异或该赛季无淘汰周数据)")
            continue
            
        avg_vote = sub['Inferred_Fan_Vote'].mean()
        avg_f_rank = sub['Fan_Vote_Rank'].mean()
        avg_j_rank = sub['Judge_Rank'].mean()
        
        # 检查规则影响
        saves = sub[(sub['Result_Actual']=='Eliminated') & (sub['Result_Counter']=='Safe')]
        kills = sub[(sub['Result_Actual']=='Safe') & (sub['Result_Counter']=='Eliminated')]
        
        impact_type = "无影响"
        impact_desc = "在两种规则下命运相同"
        
        if not saves.empty:
            impact_type = "原规则受害者"
            impact_desc = f"反事实规则可救命 (Week {saves['Week'].tolist()})"
        elif not kills.empty:
            impact_type = "原规则受益者"
            impact_desc = f"反事实规则致命 (Week {kills['Week'].tolist()})"
            
        print(f"> {t_name} (S{t_season})")
        print(f"  - 粉丝地位: Rank {avg_f_rank:.1f} ({avg_vote:.1%}) | 评委地位: Rank {avg_j_rank:.1f}")
        print(f"  - 结论: {impact_type} - {impact_desc}")
        
        controversy_summary.append({
            'Name': t_name,
            'Season': t_season,
            'Avg_Fan_Rank': avg_f_rank,
            'Impact_Type': impact_type,
            'Details': impact_desc
        })

    # 保存争议人物摘要
    pd.DataFrame(controversy_summary).to_csv('extended_controversy_analysis.csv', index=False)
    print("\n详细数据已保存至 full_data_analysis.csv 和 extended_controversy_analysis.csv")

if __name__ == "__main__":
    run_full_analysis()