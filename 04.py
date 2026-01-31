import pandas as pd
import numpy as np

# ---------------------- 1. 动态权重函数（核心修正：后期裁判权重高，第一周粉丝60%） ----------------------
def calculate_dynamic_weights(t, T):
    """
    计算第t周的粉丝权重 & 裁判权重（符合需求：前期粉丝高，后期裁判高，t=1时粉丝=0.6）
    """
    t0 = T / 2  # 拐点周数（赛季中段）
    if T <= 2:  # 边界情况：总周数≤2时固定权重
        k = 0
    else:
        # 反推k值：保证t=1时粉丝权重=0.6（趋势反转后）
        numerator = np.log(1.5)  # 由0.6 = 1/(1+exp(-k*(1-t0)))推导（趋势反转后的公式）
        denominator = t0 - 1
        k = numerator / denominator
    
    # 指数部分反转符号 → 实现“t增大→粉丝权重降低、裁判权重升高”
    exponent = k * (t - t0)  # 原公式是 -k*(t-t0)，此处去掉负号反转趋势
    w_fan = 1 / (1 + np.exp(exponent))  # 反转后的Sigmoid，保证后期粉丝权重低
    # 强制第一周粉丝权重=0.6，避免浮点误差
    if t == 1:
        w_fan = 0.6
    w_judge = 1 - w_fan
    return round(w_fan, 3), round(w_judge, 3)

# ---------------------- 2. 数据预处理（仅适配权重函数，无核心修改） ----------------------
def preprocess_data(dwts_path, fan_votes_path):
    """预处理数据：合并+标准化+划分前期/后期（整数周）"""
    dwts_data = pd.read_csv(dwts_path)
    fan_votes = pd.read_csv(fan_votes_path)

    # 裁判分宽表转长表
    score_cols = [col for col in dwts_data.columns if col.startswith("total_score_w")]
    dwts_long = dwts_data.melt(
        id_vars=["season", "celebrity_name", "eliminated_week"],
        value_vars=score_cols,
        var_name="week_col",
        value_name="judge_score"
    )
    dwts_long["week"] = dwts_long["week_col"].str.extract("(\d+)").astype(int)
    dwts_long = dwts_long.drop("week_col", axis=1).dropna(subset=["judge_score"])

    # 合并粉丝票数据
    merged_df = pd.merge(
        dwts_long,
        fan_votes[["season", "week", "contestant", "mean_fan_vote"]],
        left_on=["season", "week", "celebrity_name"],
        right_on=["season", "week", "contestant"],
        how="inner"
    ).rename(columns={"mean_fan_vote": "fan_vote"})

    # 计算赛季总周数T
    season_T = merged_df.groupby("season")["week"].max().reset_index()
    season_T.columns = ["season", "T"]
    merged_df = merged_df.merge(season_T, on="season", how="left")

    # 标准化裁判分/粉丝票（0-1区间）
    def normalize_series(s):
        return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0
    merged_df["judge_score_norm"] = merged_df.groupby(["season", "week"])["judge_score"].transform(normalize_series)
    merged_df["fan_vote_norm"] = merged_df.groupby(["season", "week"])["fan_vote"].transform(normalize_series)

    # 优化前期/后期划分（整数周）
    merged_df["t0_int"] = merged_df["T"].astype(int) // 2
    merged_df["stage"] = merged_df.apply(
        lambda x: "前期" if x["week"] <= x["t0_int"] else "后期",
        axis=1
    )

    # 过滤最后一周（决赛无淘汰）
    merged_df = merged_df[merged_df["week"] < merged_df["T"]]

    return merged_df

# ---------------------- 3. 逆转率计算（逻辑不变，适配新权重） ----------------------
def calculate_reversal_rate(merged_df):
    """
    逆转率定义（贴合规则）：
    - 裁判逆转率：裁判分倒数第一，但综合加权排名≥晋级线（晋级）的比例
    - 粉丝逆转率：粉丝票倒数第一，但综合加权排名≥晋级线（晋级）的比例
    """
    results = []

    for (season, week), group in merged_df.groupby(["season", "week"]):
        T = group["T"].iloc[0]
        w_fan, w_judge = calculate_dynamic_weights(week, T)

        # 计算综合得分（加权求和）
        group["composite_score"] = (group["judge_score_norm"] * w_judge) + (group["fan_vote_norm"] * w_fan)

        # 排名（降序：得分越高排名越前）
        group["judge_rank"] = group["judge_score_norm"].rank(ascending=False, method="min")
        group["fan_rank"] = group["fan_vote_norm"].rank(ascending=False, method="min")
        group["composite_rank"] = group["composite_score"].rank(ascending=False, method="min")

        # 晋级线：每周淘汰1人（排名最后=淘汰）
        total_players = len(group)
        elimination_rank = total_players
        group["advanced"] = group["composite_rank"] < elimination_rank

        # 统计逆转情况
        judge_bottom = group[group["judge_rank"] == total_players]
        judge_reversal_cnt = judge_bottom["advanced"].sum()
        judge_bottom_total = len(judge_bottom)

        fan_bottom = group[group["fan_rank"] == total_players]
        fan_reversal_cnt = fan_bottom["advanced"].sum()
        fan_bottom_total = len(fan_bottom)

        results.append({
            "season": season,
            "week": week,
            "stage": group["stage"].iloc[0],
            "judge_bottom_total": judge_bottom_total,
            "judge_reversal_cnt": judge_reversal_cnt,
            "fan_bottom_total": fan_bottom_total,
            "fan_reversal_cnt": fan_reversal_cnt
        })

    # 汇总前期/后期
    results_df = pd.DataFrame(results)
    stage_stats = results_df.groupby("stage").agg({
        "judge_bottom_total": "sum",
        "judge_reversal_cnt": "sum",
        "fan_bottom_total": "sum",
        "fan_reversal_cnt": "sum"
    }).reset_index()

    # 计算逆转率（避免除以0）
    stage_stats["裁判逆转率"] = np.where(
        stage_stats["judge_bottom_total"] > 0,
        (stage_stats["judge_reversal_cnt"] / stage_stats["judge_bottom_total"]).round(4),
        0.0
    )
    stage_stats["粉丝逆转率"] = np.where(
        stage_stats["fan_bottom_total"] > 0,
        (stage_stats["fan_reversal_cnt"] / stage_stats["fan_bottom_total"]).round(4),
        0.0
    )

    return stage_stats[["stage", "裁判逆转率", "粉丝逆转率"]]

# ---------------------- 4. 主流程+权重趋势验证 ----------------------
if __name__ == "__main__":
    DWTS_DATA_PATH = "cleaned_DWTS_data.csv"
    FAN_VOTES_PATH = "fan_votes_inference_with_certainty.csv"

    # 预处理
    merged_data = preprocess_data(DWTS_DATA_PATH, FAN_VOTES_PATH)
    print("数据预处理完成，有效样本量：", len(merged_data))

    # 验证权重趋势（以T=7为例，看各周权重变化）
    print("\n=== 权重趋势验证（赛季总周数=7）===")
    for t in range(1, 7):  # 周1-6（最后一周已过滤）
        w_fan, w_judge = calculate_dynamic_weights(t, 7)
        print(f"第{t}周 → 粉丝权重：{w_fan}，裁判权重：{w_judge}")

    # 计算修正后的逆转率
    reversal_result = calculate_reversal_rate(merged_data)
    print("\n=== 修正后前期/后期逆转率统计 ===")
    print(reversal_result)