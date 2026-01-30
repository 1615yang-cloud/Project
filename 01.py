import pandas as pd
import numpy as np
import scipy.stats as stats

def calculate_scores(judges_scores, fan_votes, method):
    """
    计算综合得分
    """
    if method == 'percentage':
        # 百分比法：评委分数占比 + 粉丝投票占比
        if np.sum(judges_scores) == 0:
            judge_share = np.zeros_like(judges_scores)
        else:
            judge_share = judges_scores / np.sum(judges_scores)
        return judge_share + fan_votes
        
    elif method == 'rank':
        # 排名积分法：分数越高积分越高 (Rank Points)
        # 例如：4名选手，最高分得4分，最低分得1分
        judge_points = stats.rankdata(judges_scores, method='min')
        fan_points = stats.rankdata(fan_votes, method='min')
        return judge_points + fan_points

def check_constraints_elimination(total_scores, fan_votes, eliminated_indices, method):
    """
    检查淘汰约束：被淘汰者的分数必须 <= 幸存者的分数
    对于排名法，如果分数相同，粉丝投票低者被淘汰。
    """
    if not eliminated_indices:
        return True
    
    n = len(total_scores)
    survivor_indices = [i for i in range(n) if i not in eliminated_indices]
    
    if not survivor_indices:
        return True
        
    # 提取分数和投票
    s_scores = total_scores[survivor_indices]
    e_scores = total_scores[eliminated_indices]
    
    # 快速检查：如果所有幸存者分数都严格大于被淘汰者，则通过
    if np.min(s_scores) > np.max(e_scores):
        return True
        
    if method == 'percentage':
        # 百分比法通常是连续值，直接比较大小
        return np.min(s_scores) >= np.max(e_scores)
    
    else:
        # 排名法需要处理同分 (Tie-breaker)
        # 约束：对于任意幸存者 S 和 被淘汰者 E：
        # Score(S) > Score(E) 或 (Score(S) == Score(E) 且 Fan(S) > Fan(E))
        
        s_votes = fan_votes[survivor_indices]
        e_votes = fan_votes[eliminated_indices]
        
        # 向量化比较所有对
        S_scores_grid = s_scores[:, np.newaxis]
        E_scores_grid = e_scores[np.newaxis, :]
        S_votes_grid = s_votes[:, np.newaxis]
        E_votes_grid = e_votes[np.newaxis, :]
        
        # 失败条件：幸存者分数低，或者分数相同但粉丝票没赢
        score_fail = S_scores_grid < E_scores_grid
        tie_fail = (S_scores_grid == E_scores_grid) & (S_votes_grid <= E_votes_grid)
        
        if np.any(score_fail | tie_fail):
            return False
        return True

def check_constraints_ranking(total_scores, fan_votes, placements, method):
    """
    检查排名约束（用于决赛）：Rank 1 Total >= Rank 2 Total >= ...
    """
    valid_indices = [i for i, p in enumerate(placements) if not np.isnan(p)]
    if len(valid_indices) < 2:
        return True
        
    # 按排名排序 (placement值越小越好)
    sorted_indices = sorted(valid_indices, key=lambda i: placements[i])
    
    for k in range(len(sorted_indices) - 1):
        better = sorted_indices[k]
        worse = sorted_indices[k+1]
        
        if total_scores[better] < total_scores[worse]:
            return False
            
        # 同分处理
        if method == 'rank' and total_scores[better] == total_scores[worse]:
             if fan_votes[better] <= fan_votes[worse]:
                 return False
    return True

# ==========================================
# 主处理流程
# ==========================================
df = pd.read_csv('cleaned_DWTS_data.csv')

# 预处理数据列
df['eliminated_week'] = pd.to_numeric(
    df['eliminated_week'].astype(str).str.extract('(\d+)')[0], errors='coerce'
)
df['placement'] = pd.to_numeric(df['placement'], errors='coerce')

results_list = []
n_samples = 2000 # 采样数量
burn_in = 500    # 预热期

seasons = sorted(df['season'].unique())

for season in seasons:
    # 1. 确定方法
    if 3 <= season <= 27:
        method = 'percentage'
    else:
        method = 'rank'
        
    print(f"Processing Season {season} (Method: {method})...")
    
    # 获取该赛季所有周次
    cols = [c for c in df.columns if c.startswith('total_score_w')]
    weeks = sorted([int(c.split('w')[1]) for c in cols])
    
    for week in weeks:
        score_col = f'total_score_w{week}'
        if score_col not in df.columns: continue
        
        # 2. 筛选当周活跃选手 (分数 > 0)
        current_df = df[df['season'] == season].copy()
        current_df = current_df[current_df[score_col] > 0].reset_index(drop=True)
        
        if current_df.empty: continue
            
        names = current_df['celebrity_name'].values
        judges_scores = current_df[score_col].values
        n_contestants = len(names)
        if n_contestants < 2: continue

        # 3. 确定约束类型
        # 检查本周是否有淘汰
        eliminated_mask = current_df['eliminated_week'] == week
        eliminated_indices = np.where(eliminated_mask)[0].tolist()
        
        constraint_type = 'none'
        if len(eliminated_indices) > 0:
            constraint_type = 'elimination'
        else:
            # 如果没有淘汰，检查是否是决赛（有最终排名）
            results_vals = current_df['results'].astype(str).values
            if any('Place' in r for r in results_vals):
                constraint_type = 'ranking'
                current_placements = current_df['placement'].values
            else:
                constraint_type = 'none'
        
        if constraint_type == 'none': continue

        # 4. MCMC 采样
        samples = []
        current_votes = np.random.dirichlet(np.ones(n_contestants))
        
        # 寻找合法的初始点
        valid_start = False
        for _ in range(1000):
            scores = calculate_scores(judges_scores, current_votes, method)
            if constraint_type == 'elimination':
                if check_constraints_elimination(scores, current_votes, eliminated_indices, method):
                    valid_start = True
                    break
            elif constraint_type == 'ranking':
                if check_constraints_ranking(scores, current_votes, current_placements, method):
                    valid_start = True
                    break
            current_votes = np.random.dirichlet(np.ones(n_contestants))
            
        if not valid_start:
            # print(f"Warning: S{season} W{week} constraints valid start not found.")
            continue
            
        # Metropolis-Hastings 循环
        for i in range(n_samples + burn_in):
            proposal = current_votes.copy()
            # 随机扰动：选取两个人交换少量票数
            idx1, idx2 = np.random.choice(n_contestants, 2, replace=False)
            step = np.random.uniform(0, 0.05)
            
            if proposal[idx1] > step:
                proposal[idx1] -= step
                proposal[idx2] += step
            else:
                proposal[idx1] += step
                proposal[idx2] -= step
            
            # 检查提议是否满足约束
            scores_prop = calculate_scores(judges_scores, proposal, method)
            is_valid = False
            if constraint_type == 'elimination':
                is_valid = check_constraints_elimination(scores_prop, proposal, eliminated_indices, method)
            elif constraint_type == 'ranking':
                is_valid = check_constraints_ranking(scores_prop, proposal, current_placements, method)
                
            if is_valid:
                current_votes = proposal
            
            if i >= burn_in:
                samples.append(current_votes.copy())
        
        # 5. 统计结果
        samples_np = np.array(samples)
        if len(samples_np) == 0: continue
            
        means = samples_np.mean(axis=0)
        lowers = np.percentile(samples_np, 2.5, axis=0)
        uppers = np.percentile(samples_np, 97.5, axis=0)
        
        for idx, name in enumerate(names):
            results_list.append({
                'season': season,
                'week': week,
                'contestant': name,
                'method': method,
                'judges_score': judges_scores[idx],
                'is_eliminated': idx in eliminated_indices,
                'mean_fan_vote': means[idx],
                'ci_lower': lowers[idx],
                'ci_upper': uppers[idx]
            })

# 保存结果
results_df = pd.DataFrame(results_list)
results_df.to_csv('fan_votes_inference_v2.csv', index=False)