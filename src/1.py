# -*- coding: utf-8 -*-
"""
MCM/ICM 2025 Problem C - Olympic Medal Table Model
只使用题目给的五个数据文件：
- data_dictionary.csv
- summerOly_athletes.csv
- summerOly_medal_counts.csv（本脚本不依赖它来训练；可用于对照核验）
- summerOly_hosts.csv
- summerOly_programs.csv

核心思路：
1) 用 athletes 直接聚合得到每个 NOC-Year 的 Gold / Total（含 0 medal）
2) 构造特征：上一届/历史 EMA、运动员规模、参赛运动数、主场、项目结构(按 sport)
3) 分别拟合 Total 与 Gold 的负二项回归（NB2）
4) 用参数协方差 + NB 计数噪声模拟 -> 预测区间
5) Time-based CV 评估误差与区间覆盖率
6) 预测 LA 2028 medal table；统计首次获牌国家数；输出各国关键项目

运行：
python mcm_medal_model.py

输出：
./outputs/
  - cv_metrics.json
  - la2028_predictions.csv
  - la2028_top_improve.csv
  - la2028_top_decline.csv
  - first_medal_countries_2028.csv
  - sport_importance_top5_by_country.csv
"""

import os
import re
import json
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# =========================
# 配置区
# =========================
DATA_DIR = "/mnt/data"
ATHLETES_PATH = os.path.join(DATA_DIR, "summerOly_athletes.csv")
HOSTS_PATH    = os.path.join(DATA_DIR, "summerOly_hosts.csv")
PROGRAMS_PATH = os.path.join(DATA_DIR, "summerOly_programs.csv")

OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 训练只用到 2024（预测 2028）
TRAIN_END_YEAR = 2024
PRED_YEAR = 2028

# 选择进入“项目结构”的 sport 数量（太多会高维且稀疏）
TOP_SPORTS_K = 15

# 预测区间：给 80% 和 95%
PI_LEVELS = [(0.10, 0.90), (0.025, 0.975)]  # (low, high)

# 模拟次数（越大区间越稳定，但会更慢）
N_SIM = 5000


# =========================
# 工具函数
# =========================
def clean_host_country(host_str: str) -> str:
    """从 'City, Country' 里提取 Country，并清洗空格/不可见字符。"""
    if pd.isna(host_str):
        return ""
    s = str(host_str).replace("\xa0", " ").strip()
    # 取最后一个逗号后的部分作为国家
    parts = [p.strip() for p in s.split(",")]
    return parts[-1] if parts else s

def nb_mu_to_nbinom_params(mu: np.ndarray, alpha: float):
    """
    statsmodels NB2 常用参数：Var(Y)=mu + alpha*mu^2
    numpy.random.negative_binomial 需要 (n, p)：
      mean = n*(1-p)/p
    令 n = 1/alpha, p = n/(n+mu) = 1/(1+alpha*mu)
    """
    alpha = max(alpha, 1e-12)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return n, p

def safe_log1p(x):
    return np.log1p(np.maximum(x, 0))

def olympic_years_from_athletes(df_ath: pd.DataFrame):
    years = sorted(df_ath["Year"].dropna().unique().tolist())
    return [int(y) for y in years if isinstance(y, (int, np.integer)) or str(y).isdigit()]


# =========================
# 1) 读数据
# =========================
def load_data():
    # athletes 很大，但可直接读；如你内存紧张可加 usecols / chunksize
    athletes = pd.read_csv(ATHLETES_PATH)
    hosts = pd.read_csv(HOSTS_PATH)
    # programs 可能不是 utf-8（题目数据经常带特殊字符）
    programs = pd.read_csv(PROGRAMS_PATH, encoding="latin1")
    return athletes, hosts, programs


# =========================
# 2) 用 athletes 聚合 medal counts（含 0 medal）
# =========================
def build_country_year_targets(athletes: pd.DataFrame) -> pd.DataFrame:
    """
    输出每个 NOC-Year：
    - total_medals: Medal != 'No medal'
    - gold_medals: Medal == 'Gold'
    - athletes_cnt: unique(Name) 作为参赛规模近似
    - sports_cnt: unique(Sport)
    - medalists_cnt: unique(Name) among medalists（可选特征）
    """
    df = athletes.copy()

    # 统一 Medal 字段
    df["Medal"] = df["Medal"].fillna("No medal")
    df["is_medal"] = (df["Medal"] != "No medal").astype(int)
    df["is_gold"]  = (df["Medal"] == "Gold").astype(int)

    # 有些 Team 会像 Germany-1（沙排/团体），但我们用 NOC 做主键即可
    # unique athlete 计数：用 Name 去重（同届同运动员理论上只出现一次；若不止一次也能近似规模）
    g = df.groupby(["NOC", "Year"], as_index=False)

    agg = g.agg(
        total_medals=("is_medal", "sum"),
        gold_medals=("is_gold", "sum"),
        athletes_cnt=("Name", pd.Series.nunique),
        sports_cnt=("Sport", pd.Series.nunique),
        medalists_cnt=("Name", lambda x: x.nunique()),  # 先占位，下面用 medalists 更准确
    )

    # 更准确的 medalists_cnt：只在获牌子样本里数 unique(Name)
    medalists = df.loc[df["is_medal"] == 1].groupby(["NOC", "Year"])["Name"].nunique()
    agg["medalists_cnt"] = agg.set_index(["NOC", "Year"]).index.map(medalists).fillna(0).astype(int)

    # 补齐所有 NOC-Year：athletes 本身就包含参赛国家与年份，groupby 后已经覆盖到“0 medal”
    return agg


# =========================
# 3) Host 主场特征（从 hosts 推断 host NOC）
# =========================
def build_host_map(hosts: pd.DataFrame, athletes: pd.DataFrame) -> dict:
    """
    hosts: Year -> Host string (City, Country)
    目标：Year -> host_noc（用 athletes 的 Team==host_country 反推最常见 NOC）
    """
    # 建立 Team(国家名) -> 最常见 NOC 的映射（全局）
    team_noc_mode = (athletes.groupby("Team")["NOC"]
                     .agg(lambda s: s.value_counts().index[0])
                     .to_dict())

    year_host_noc = {}
    for _, r in hosts.iterrows():
        y = int(r["Year"])
        country = clean_host_country(r["Host"])
        # 优先用该国家名直接映射
        noc = team_noc_mode.get(country, None)

        # 若没找到，尝试更宽松匹配（去掉括号内容/多余空格）
        if noc is None:
            country2 = re.sub(r"\(.*?\)", "", country).strip()
            noc = team_noc_mode.get(country2, None)

        # 最后兜底：常见国家名手写映射（可按需要补）
        if noc is None:
            fallback = {
                "United States": "USA",
                "United Kingdom": "GBR",
                "Soviet Union": "URS",
                "Russia": "RUS",
                "Korea, South": "KOR",
                "China": "CHN",
                "Japan": "JPN",
                "France": "FRA",
                "Germany": "GER",
                "Australia": "AUS",
                "Italy": "ITA",
                "Spain": "ESP",
                "Brazil": "BRA",
                "Canada": "CAN",
                "Greece": "GRE",
            }
            noc = fallback.get(country, "")

        year_host_noc[y] = noc if noc is not None else ""

    return year_host_noc


# =========================
# 4) Programs -> 每年每个 Sport 的 event counts（若缺 2028，则用 2024 近似）
# =========================
def build_events_by_sport_year(programs: pd.DataFrame) -> pd.DataFrame:
    """
    programs 结构：每行是 Discipline，含 Sport 列 + 多个年份列（字符串年份）
    我们按 Sport 汇总 -> (Sport, Year) 的 events 数
    """
    df = programs.copy()

    # 年份列可能有 '1906*'，把 * 去掉
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}\*?", str(c).strip())]
    long = df.melt(
        id_vars=["Sport"],
        value_vars=year_cols,
        var_name="Year",
        value_name="events"
    )
    long["Year"] = long["Year"].astype(str).str.replace("*", "", regex=False).astype(int)
    long["events"] = pd.to_numeric(long["events"], errors="coerce").fillna(0.0)

    # 同一 Sport 多 Discipline，求和
    out = (long.groupby(["Sport", "Year"], as_index=False)["events"].sum())
    return out


def get_year_events_matrix(events_long: pd.DataFrame, target_year: int) -> pd.Series:
    """
    返回某一年：Sport -> event_count 的 Series
    若 target_year 不在数据里，则用最新一年近似（一般是 2024）
    """
    years = sorted(events_long["Year"].unique().tolist())
    if target_year not in years:
        ref_year = max(years)
    else:
        ref_year = target_year

    s = (events_long.loc[events_long["Year"] == ref_year]
         .set_index("Sport")["events"]
         .sort_values(ascending=False))
    return s


# =========================
# 5) sport-strength 特征：国家在各 sport 的历史获牌倾向
# =========================
def build_sport_strength_features(athletes: pd.DataFrame,
                                 events_long: pd.DataFrame,
                                 top_sports: list,
                                 k_lag: int = 3) -> pd.DataFrame:
    """
    输出每个 NOC-Year 的 sport strength（滚动历史）：
      strength_total_{sport} = 最近 k_lag 届(不含本届)该 sport medals / 最近 k_lag 届总 medals
    然后在最终特征里与该年的 events_{sport} 做交互： strength * events
    """
    df = athletes.copy()
    df["Medal"] = df["Medal"].fillna("No medal")
    df["is_medal"] = (df["Medal"] != "No medal").astype(int)
    df["is_gold"] = (df["Medal"] == "Gold").astype(int)

    # 每国每年每 sport medals
    cs = (df.groupby(["NOC", "Year", "Sport"], as_index=False)
            .agg(medals_sport=("is_medal", "sum"),
                 gold_sport=("is_gold", "sum")))

    # 每国每年总 medals
    cy = (df.groupby(["NOC", "Year"], as_index=False)
            .agg(medals_total=("is_medal", "sum"),
                 gold_total=("is_gold", "sum")))

    # 只保留 top sports
    cs = cs[cs["Sport"].isin(top_sports)].copy()

    # 变宽：每 sport 一列
    wide_total = cs.pivot_table(index=["NOC", "Year"], columns="Sport", values="medals_sport", fill_value=0)
    wide_gold  = cs.pivot_table(index=["NOC", "Year"], columns="Sport", values="gold_sport", fill_value=0)

    # 对齐年份顺序
    wide_total = wide_total.sort_index()
    wide_gold  = wide_gold.sort_index()

    # 把 cy 也变成 index 对齐
    cy_idx = cy.set_index(["NOC", "Year"]).sort_index()

    # 计算 rolling sums（按 NOC 分组滚动）
    # 注意：奥运不是严格每 4 年且历史存在缺口，因此这里按“届次顺序”滚动更合理
    def rolling_sum_by_group(mat: pd.DataFrame, window: int):
        out = []
        for noc, sub in mat.groupby(level=0):
            sub2 = sub.droplevel(0)
            rs = sub2.rolling(window=window, min_periods=1).sum().shift(1)  # shift(1) 排除本届
            rs.index = pd.MultiIndex.from_product([[noc], rs.index], names=["NOC", "Year"])
            out.append(rs)
        return pd.concat(out).sort_index()

    # 最近 k_lag 届 sport medals 累计（不含本届）
    roll_total = rolling_sum_by_group(wide_total, k_lag)
    roll_gold  = rolling_sum_by_group(wide_gold,  k_lag)

    # 最近 k_lag 届总 medals 累计（不含本届）
    # 先取 cy_idx，再 rolling
    cy_med = cy_idx[["medals_total", "gold_total"]].copy()
    cy_med = cy_med.sort_index()

    def rolling_sum_series(df2: pd.DataFrame, col: str, window: int):
        out = []
        for noc, sub in df2.groupby(level=0):
            sub2 = sub.droplevel(0)
            rs = sub2[col].rolling(window=window, min_periods=1).sum().shift(1)
            rs.index = pd.MultiIndex.from_product([[noc], rs.index], names=["NOC", "Year"])
            out.append(rs)
        return pd.concat(out).sort_index()

    roll_medals_total = rolling_sum_series(cy_med, "medals_total", k_lag)
    roll_gold_total   = rolling_sum_series(cy_med, "gold_total",   k_lag)

    # strength = sport_roll / total_roll（避免除 0）
    strength_total = roll_total.div(roll_medals_total.replace(0, np.nan), axis=0).fillna(0.0)
    strength_gold  = roll_gold.div(roll_gold_total.replace(0, np.nan), axis=0).fillna(0.0)

    # 加前缀列名
    strength_total.columns = [f"strength_total__{c}" for c in strength_total.columns]
    strength_gold.columns  = [f"strength_gold__{c}" for c in strength_gold.columns]

    out = pd.concat([strength_total, strength_gold], axis=1).reset_index()
    return out


# =========================
# 6) 构造最终训练表
# =========================
def build_model_table(athletes: pd.DataFrame, hosts: pd.DataFrame, programs: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    返回：
    - full_table: NOC-Year 的特征 + (gold_medals, total_medals)
    - top_sports: 进入模型的 sport 列表
    """
    # targets + 基础参赛规模特征
    cy = build_country_year_targets(athletes)

    # host
    year_host_noc = build_host_map(hosts, athletes)
    cy["is_host"] = cy["Year"].map(lambda y: 1 if year_host_noc.get(int(y), "") == cy.loc[cy["Year"] == y, "NOC"].iloc[0] else 0)
    # 上面写法不够稳（依赖 cy.loc），改成更直接：
    cy["host_noc"] = cy["Year"].map(lambda y: year_host_noc.get(int(y), ""))
    cy["is_host"] = (cy["NOC"] == cy["host_noc"]).astype(int)

    # events by sport-year
    ev_long = build_events_by_sport_year(programs)
    # 计算 top sports（按历史总 events 或历史总 medals都行；这里用 medals 更贴近“产牌”）
    df = athletes.copy()
    df["Medal"] = df["Medal"].fillna("No medal")
    df["is_medal"] = (df["Medal"] != "No medal").astype(int)
    sport_medals = (df.groupby("Sport")["is_medal"].sum().sort_values(ascending=False))
    top_sports = sport_medals.head(TOP_SPORTS_K).index.tolist()

    # sport strength（历史滚动）
    strength = build_sport_strength_features(athletes, ev_long, top_sports, k_lag=3)

    # 合并
    table = cy.merge(strength, on=["NOC", "Year"], how="left").fillna(0.0)

    # 事件数量：加入总 events 以及每个 sport 的 events
    # 若某 year 不存在（如 2028），后续预测时再处理
    ev_pivot = ev_long.pivot_table(index="Year", columns="Sport", values="events", fill_value=0.0)
    ev_pivot["events_total_all_sports"] = ev_pivot.sum(axis=1)
    ev_pivot = ev_pivot.reset_index()

    table = table.merge(ev_pivot[["Year", "events_total_all_sports"] + [s for s in top_sports if s in ev_pivot.columns]],
                        on="Year", how="left").fillna(0.0)

    # 构造交互项： strength * events_sport
    for s in top_sports:
        if s not in table.columns:
            table[s] = 0.0
        st = f"strength_total__{s}"
        sg = f"strength_gold__{s}"
        table[f"int_total__{s}"] = table[st] * table[s]
        table[f"int_gold__{s}"]  = table[sg] * table[s]

    # 追加：滞后/EMA 特征（更强的时间序列信号）
    table = table.sort_values(["NOC", "Year"]).reset_index(drop=True)

    def add_lag_ema(df_in: pd.DataFrame, col: str, ema_span: int = 3):
        out = df_in.copy()
        out[f"lag1_{col}"] = out.groupby("NOC")[col].shift(1).fillna(0.0)
        # 按“届次序列”做 EMA（不严格按 4 年）
        out[f"ema_{col}"] = (out.groupby("NOC")[col]
                               .apply(lambda s: s.shift(1).ewm(span=ema_span, adjust=False).mean())
                               .reset_index(level=0, drop=True)
                               .fillna(0.0))
        return out

    table = add_lag_ema(table, "total_medals", ema_span=3)
    table = add_lag_ema(table, "gold_medals",  ema_span=3)
    table = add_lag_ema(table, "athletes_cnt", ema_span=3)
    table = add_lag_ema(table, "sports_cnt",   ema_span=3)

    # log 变换：规模类更线性
    for c in ["athletes_cnt", "lag1_athletes_cnt", "ema_athletes_cnt",
              "sports_cnt", "lag1_sports_cnt", "ema_sports_cnt"]:
        table[f"log1p_{c}"] = safe_log1p(table[c].values)

    # 训练范围：去掉 1906（如有），并只留到 TRAIN_END_YEAR
    table = table[(table["Year"] <= TRAIN_END_YEAR) & (table["Year"] != 1906)].copy()

    return table, top_sports


# =========================
# 7) 拟合 Negative Binomial（离散模型，估 alpha）
# =========================
def fit_nb_model(df: pd.DataFrame, y_col: str, x_cols: list):
    """
    使用 statsmodels.discrete.NegativeBinomial 拟合：
      E[y|x] = exp(X beta)
    """
    X = df[x_cols].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df[y_col].astype(float).values

    model = sm.NegativeBinomial(y, X)
    res = model.fit(disp=False, maxiter=200)

    return res


def prepare_design(df: pd.DataFrame, x_cols: list, scaler: StandardScaler = None, fit_scaler: bool = True):
    """
    标准化（不标准化也行，但标准化更稳）
    """
    X = df[x_cols].astype(float).values
    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True)
    if fit_scaler:
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)
    Xs = pd.DataFrame(Xs, columns=x_cols, index=df.index)
    return Xs, scaler


# =========================
# 8) Time-based CV 评估
# =========================
def time_based_cv(df: pd.DataFrame, y_col: str, x_cols: list, test_years: list):
    metrics = []
    cover = {str(l): [] for l in PI_LEVELS}

    for ty in test_years:
        train = df[df["Year"] < ty].copy()
        test  = df[df["Year"] == ty].copy()
        if len(test) == 0 or len(train) == 0:
            continue

        X_train_s, scaler = prepare_design(train, x_cols, scaler=None, fit_scaler=True)
        X_test_s, _ = prepare_design(test, x_cols, scaler=scaler, fit_scaler=False)

        # 拟合 NB：用标准化后的 X
        train2 = train.copy()
        test2 = test.copy()
        for c in x_cols:
            train2[c] = X_train_s[c]
            test2[c]  = X_test_s[c]

        res = fit_nb_model(train2, y_col=y_col, x_cols=x_cols)

        # 点预测
        Xp = sm.add_constant(test2[x_cols], has_constant="add")
        mu = res.predict(Xp)

        y_true = test2[y_col].values
        mae = float(np.mean(np.abs(y_true - mu)))
        rmse = float(np.sqrt(np.mean((y_true - mu) ** 2)))

        # 预测区间覆盖率（用快速模拟，次数少一点）
        alpha = float(res.params.get("alpha", getattr(res, "lnalpha", 0.0)))
        if "alpha" not in res.params.index:
            # 有些版本 alpha 不在 params；用 res.model._dispersion 或 lnalpha 近似
            try:
                alpha = float(np.exp(res.params["lnalpha"]))
            except Exception:
                alpha = 0.2  # 兜底

        cov = res.cov_params()

        # 系数抽样：多元正态（参数不确定性）
        beta_mean = res.params.values
        n_draw = 1200
        beta_draw = np.random.multivariate_normal(beta_mean, cov, size=n_draw)

        Xmat = Xp.values  # (n, p)
        # 线性预测 -> mu_draw
        eta_draw = beta_draw @ Xmat.T  # (n_draw, n)
        mu_draw = np.exp(eta_draw)

        # 加计数噪声
        n_param, p_param = nb_mu_to_nbinom_params(mu_draw, alpha)
        y_sim = np.random.negative_binomial(n=n_param, p=p_param).astype(float)

        for (lo, hi) in PI_LEVELS:
            qlo = np.quantile(y_sim, lo, axis=0)
            qhi = np.quantile(y_sim, hi, axis=0)
            cov_rate = float(np.mean((y_true >= qlo) & (y_true <= qhi)))
            cover[str((lo, hi))].append(cov_rate)

        metrics.append({
            "test_year": int(ty),
            "n_test": int(len(test2)),
            "MAE": mae,
            "RMSE": rmse
        })

    out = {
        "y_col": y_col,
        "folds": metrics,
        "MAE_mean": float(np.mean([m["MAE"] for m in metrics])) if metrics else None,
        "RMSE_mean": float(np.mean([m["RMSE"] for m in metrics])) if metrics else None,
        "PI_coverage_mean": {k: float(np.mean(v)) if len(v) else None for k, v in cover.items()}
    }
    return out


# =========================
# 9) 全量训练 + 2028 预测（含预测区间）
# =========================
def fit_and_predict_2028(full_df: pd.DataFrame, top_sports: list):
    """
    返回：
    - pred_df: 每个 NOC 的 2028 gold/total 点预测 + 区间
    - sim_store: (optional) 用于后续首次获牌/概率分析
    """
    # 选择特征列
    base_x = [
        "is_host",
        "events_total_all_sports",
        "log1p_athletes_cnt",
        "log1p_lag1_athletes_cnt",
        "log1p_sports_cnt",
        "lag1_total_medals", "ema_total_medals",
        "lag1_gold_medals",  "ema_gold_medals",
    ]
    # 加入交互项（total/gold 分开）
    x_total = base_x + [f"int_total__{s}" for s in top_sports]
    x_gold  = base_x + [f"int_gold__{s}"  for s in top_sports]

    # 训练数据
    train = full_df.copy()

    # 构造 2028 行：用 2024 的最后一届特征作基础（对每个 NOC 做 shift）
    # 逻辑：预测下一届的国家状态 = 已知到 2024 的滞后/EMA/规模等
    latest = train.sort_values(["NOC", "Year"]).groupby("NOC").tail(1).copy()
    pred = latest.copy()
    pred["Year"] = PRED_YEAR

    # 主场：从 hosts 推断 2028 host_noc（通常 USA）
    # 这里直接用 hosts 文件里 2028 -> host country -> host noc 的流程：
    # 为避免重复建 map，这里简单：如果 NOC==USA，则 is_host=1（你可按需改成更通用）
    pred["is_host"] = (pred["NOC"] == "USA").astype(int)

    # events_total_all_sports：若 programs 无 2028，则用 2024 近似
    # 由于 full_df 只到 2024，这里用 2024 的全局 events_total_all_sports
    events_2024 = float(train.loc[train["Year"] == 2024, "events_total_all_sports"].iloc[0]) \
        if (train["Year"] == 2024).any() else float(train["events_total_all_sports"].max())
    pred["events_total_all_sports"] = events_2024

    # 各 sport 的 events：同理用 2024 近似（也支持你手动修改某些 sport）
    # ======= 你如果知道 2028 项目调整，可在这里填 =======
    manual_event_overrides_2028 = {
        # "Swimming": 37,
        # "Athletics": 50,
    }
    # =======================================================
    for s in top_sports:
        # pred 里存着 2024 的 events(sport) 列（因为 latest 来自 full_df）
        if s in manual_event_overrides_2028:
            pred[s] = float(manual_event_overrides_2028[s])
        else:
            # 用 2024 的该 sport events（若缺则 0）
            if s in pred.columns:
                pred[s] = pred[s].astype(float)
            else:
                pred[s] = 0.0

    # 重新计算交互项（因为 events 可能被 override）
    for s in top_sports:
        pred[f"int_total__{s}"] = pred.get(f"strength_total__{s}", 0.0) * pred[s]
        pred[f"int_gold__{s}"]  = pred.get(f"strength_gold__{s}",  0.0) * pred[s]

    # 标准化（分别对 total/gold 的 X）
    X_train_total_s, scaler_total = prepare_design(train, x_total, scaler=None, fit_scaler=True)
    X_train_gold_s,  scaler_gold  = prepare_design(train, x_gold,  scaler=None, fit_scaler=True)

    train_total = train.copy()
    train_gold  = train.copy()
    for c in x_total:
        train_total[c] = X_train_total_s[c]
    for c in x_gold:
        train_gold[c] = X_train_gold_s[c]

    # 拟合两个 NB
    res_total = fit_nb_model(train_total, y_col="total_medals", x_cols=x_total)
    res_gold  = fit_nb_model(train_gold,  y_col="gold_medals",  x_cols=x_gold)

    # 预测集标准化
    X_pred_total_s, _ = prepare_design(pred, x_total, scaler=scaler_total, fit_scaler=False)
    X_pred_gold_s,  _ = prepare_design(pred, x_gold,  scaler=scaler_gold,  fit_scaler=False)

    pred_total = pred.copy()
    pred_gold  = pred.copy()
    for c in x_total:
        pred_total[c] = X_pred_total_s[c]
    for c in x_gold:
        pred_gold[c] = X_pred_gold_s[c]

    Xp_total = sm.add_constant(pred_total[x_total], has_constant="add")
    Xp_gold  = sm.add_constant(pred_gold[x_gold],  has_constant="add")

    mu_total = res_total.predict(Xp_total)
    mu_gold  = res_gold.predict(Xp_gold)

    # ---- 模拟区间：参数不确定性 + NB 计数噪声 ----
    def simulate_pi(res, Xp, mu_point):
        # alpha
        try:
            alpha = float(np.exp(res.params["lnalpha"]))
        except Exception:
            alpha = float(res.params["alpha"]) if "alpha" in res.params.index else 0.2

        cov = res.cov_params()
        beta_mean = res.params.values
        beta_draw = np.random.multivariate_normal(beta_mean, cov, size=N_SIM)  # (N_SIM, p)

        Xmat = Xp.values  # (n, p)
        eta_draw = beta_draw @ Xmat.T  # (N_SIM, n)
        mu_draw = np.exp(eta_draw)

        n_param, p_param = nb_mu_to_nbinom_params(mu_draw, alpha)
        y_sim = np.random.negative_binomial(n=n_param, p=p_param).astype(float)  # (N_SIM, n)

        # 汇总区间
        out = {
            "mu_point": np.asarray(mu_point).astype(float),
            "alpha": float(alpha),
            "y_sim": y_sim
        }
        for (lo, hi) in PI_LEVELS:
            out[f"pi_{int(lo*1000)}_{int(hi*1000)}_lo"] = np.quantile(y_sim, lo, axis=0)
            out[f"pi_{int(lo*1000)}_{int(hi*1000)}_hi"] = np.quantile(y_sim, hi, axis=0)
        return out

    sim_total = simulate_pi(res_total, Xp_total, mu_total)
    sim_gold  = simulate_pi(res_gold,  Xp_gold,  mu_gold)

    # 组装输出表
    out = pd.DataFrame({
        "NOC": pred["NOC"].values,
        "Year": PRED_YEAR,
        "total_pred": sim_total["mu_point"],
        "gold_pred":  sim_gold["mu_point"],
    })

    # 加区间
    for (lo, hi) in PI_LEVELS:
        key_lo = f"pi_{int(lo*1000)}_{int(hi*1000)}_lo"
        key_hi = f"pi_{int(lo*1000)}_{int(hi*1000)}_hi"
        out[f"total_{int(lo*100)}_{int(hi*100)}_lo"] = sim_total[key_lo]
        out[f"total_{int(lo*100)}_{int(hi*100)}_hi"] = sim_total[key_hi]
        out[f"gold_{int(lo*100)}_{int(hi*100)}_lo"]  = sim_gold[key_lo]
        out[f"gold_{int(lo*100)}_{int(hi*100)}_hi"]  = sim_gold[key_hi]

    # 四舍五入更像 medal table（但保留原值也行）
    for c in out.columns:
        if c not in ["NOC", "Year"]:
            out[c] = out[c].astype(float)

    # 同时返回模拟矩阵用于“首次获牌概率”等分析
    sim_store = {
        "total": sim_total["y_sim"],  # (N_SIM, n_countries)
        "gold":  sim_gold["y_sim"],
        "noc_order": out["NOC"].tolist()
    }

    return out, sim_store, res_total, res_gold, x_total, x_gold


# =========================
# 10) 首次获牌国家数预测（概率）
# =========================
def first_medal_analysis(train_df: pd.DataFrame, pred_df: pd.DataFrame, sim_store: dict):
    """
    找出历史到 2024 从未拿过 medal 的 NOC：
    - 给出其在 2028 total>=1 的概率
    - 估计“首次获牌国家数”的期望与分布
    """
    hist = train_df.groupby("NOC")["total_medals"].sum()
    never_medal_nocs = hist[hist == 0].index.tolist()

    nocs = sim_store["noc_order"]
    idx = [nocs.index(n) for n in never_medal_nocs if n in nocs]
    if len(idx) == 0:
        return None, None

    ysim_total = sim_store["total"][:, idx]  # (N_SIM, n_never)
    prob_each = (ysim_total >= 1).mean(axis=0)

    per_country = pd.DataFrame({
        "NOC": [never_medal_nocs[i] for i in range(len(never_medal_nocs)) if never_medal_nocs[i] in nocs],
        "P_first_medal_2028": prob_each
    }).sort_values("P_first_medal_2028", ascending=False)

    # 首次获牌国家数分布
    first_count = (ysim_total >= 1).sum(axis=1)  # 每次模拟里有多少国家首次获牌
    expected = float(first_count.mean())
    # 给一个“最可能值”(mode 近似用频数最高)
    vals, cnts = np.unique(first_count, return_counts=True)
    mode_val = int(vals[np.argmax(cnts)])
    # 给区间（例如 80% 与 95%）
    dist_summary = {
        "expected_first_medal_countries": expected,
        "mode_first_medal_countries": mode_val,
        "P_first_medal_countries_equals_mode": float(cnts.max() / cnts.sum()),
        "PI_80": [int(np.quantile(first_count, 0.10)), int(np.quantile(first_count, 0.90))],
        "PI_95": [int(np.quantile(first_count, 0.025)), int(np.quantile(first_count, 0.975))]
    }

    return per_country, dist_summary


# =========================
# 11) 各国最重要 sport（按模型贡献度近似）
# =========================
def sport_importance(res_total, res_gold, x_total, x_gold, pred_feature_df: pd.DataFrame, top_sports: list):
    """
    用线性预测项里的系数 * 特征值，近似每个 sport 对该国的“贡献度”。
    注意：这是解释性指标（log-link 下是对 log(mu) 的加性贡献），用于排序很合适。
    """
    # 提取系数
    bt = res_total.params
    bg = res_gold.params

    rows = []
    for _, r in pred_feature_df.iterrows():
        noc = r["NOC"]
        # 对每个 sport 取交互项
        contrib = []
        for s in top_sports:
            ft = f"int_total__{s}"
            fg = f"int_gold__{s}"
            ct = float(bt.get(ft, 0.0)) * float(r.get(ft, 0.0))
            cg = float(bg.get(fg, 0.0)) * float(r.get(fg, 0.0))
            contrib.append((s, ct, cg))

        # 排序：total 贡献为主
        contrib.sort(key=lambda x: x[1], reverse=True)
        top5 = contrib[:5]
        for rank, (s, ct, cg) in enumerate(top5, start=1):
            rows.append({
                "NOC": noc,
                "rank": rank,
                "sport": s,
                "log_mu_contrib_total": ct,
                "log_mu_contrib_gold": cg
            })

    return pd.DataFrame(rows)


# =========================
# 主流程
# =========================
def main():
    athletes, hosts, programs = load_data()

    # 建表
    full_table, top_sports = build_model_table(athletes, hosts, programs)

    # CV（挑一些较现代的年份，更贴近 2028 预测）
    years = sorted(full_table["Year"].unique().tolist())
    test_years = [y for y in years if y >= 2000]  # 2000-2024
    # 只评估到 2024
    test_years = [y for y in test_years if y <= TRAIN_END_YEAR]

    # 特征列会在 fit_and_predict_2028 里定义；这里为了 CV 复用一次构造
    # （保持一致）
    base_x = [
        "is_host",
        "events_total_all_sports",
        "log1p_athletes_cnt",
        "log1p_lag1_athletes_cnt",
        "log1p_sports_cnt",
        "lag1_total_medals", "ema_total_medals",
        "lag1_gold_medals",  "ema_gold_medals",
    ]
    x_total = base_x + [f"int_total__{s}" for s in top_sports]
    x_gold  = base_x + [f"int_gold__{s}"  for s in top_sports]

    cv_total = time_based_cv(full_table, "total_medals", x_total, test_years)
    cv_gold  = time_based_cv(full_table, "gold_medals",  x_gold,  test_years)

    cv_metrics = {"total": cv_total, "gold": cv_gold}
    with open(os.path.join(OUT_DIR, "cv_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(cv_metrics, f, ensure_ascii=False, indent=2)

    # 训练并预测 2028
    pred_df, sim_store, res_total, res_gold, x_total, x_gold = fit_and_predict_2028(full_table, top_sports)

    # medal table 排序：先 gold 再 total（你也可以按 total 排）
    pred_df_sorted = pred_df.sort_values(["gold_pred", "total_pred"], ascending=False).reset_index(drop=True)
    pred_df_sorted.to_csv(os.path.join(OUT_DIR, "la2028_predictions.csv"), index=False)

    # 相对 2024 的提升/下滑（点预测）
    actual_2024 = full_table[full_table["Year"] == 2024][["NOC", "gold_medals", "total_medals"]].copy()
    comp = pred_df.merge(actual_2024, on="NOC", how="left").fillna(0.0)
    comp["delta_total"] = comp["total_pred"] - comp["total_medals"]
    comp["delta_gold"]  = comp["gold_pred"]  - comp["gold_medals"]

    improve = comp.sort_values("delta_total", ascending=False).head(20)
    decline = comp.sort_values("delta_total", ascending=True).head(20)
    improve.to_csv(os.path.join(OUT_DIR, "la2028_top_improve.csv"), index=False)
    decline.to_csv(os.path.join(OUT_DIR, "la2028_top_decline.csv"), index=False)

    # 首次获牌分析
    per_country, dist_summary = first_medal_analysis(full_table, pred_df, sim_store)
    if per_country is not None:
        per_country.to_csv(os.path.join(OUT_DIR, "first_medal_countries_2028.csv"), index=False)
        with open(os.path.join(OUT_DIR, "first_medal_summary_2028.json"), "w", encoding="utf-8") as f:
            json.dump(dist_summary, f, ensure_ascii=False, indent=2)

    # sport importance（需要用于 2028 的特征表：这里用 fit_and_predict_2028 的 pred 内部结构）
    # 为了复用，我们重建一个“预测特征表”：取 full_table 每国最后一届并设置 Year=2028，逻辑同上
    latest = full_table.sort_values(["NOC", "Year"]).groupby("NOC").tail(1).copy()
    pred_feat = latest.copy()
    pred_feat["Year"] = PRED_YEAR
    pred_feat["is_host"] = (pred_feat["NOC"] == "USA").astype(int)
    events_2024 = float(full_table.loc[full_table["Year"] == 2024, "events_total_all_sports"].iloc[0]) \
        if (full_table["Year"] == 2024).any() else float(full_table["events_total_all_sports"].max())
    pred_feat["events_total_all_sports"] = events_2024

    # 用 2024 events 近似 2028（可手动 override）
    manual_event_overrides_2028 = {}
    for s in top_sports:
        if s in manual_event_overrides_2028:
            pred_feat[s] = float(manual_event_overrides_2028[s])

    # 更新交互项
    for s in top_sports:
        pred_feat[f"int_total__{s}"] = pred_feat.get(f"strength_total__{s}", 0.0) * pred_feat.get(s, 0.0)
        pred_feat[f"int_gold__{s}"]  = pred_feat.get(f"strength_gold__{s}",  0.0) * pred_feat.get(s, 0.0)

    # 重要：sport_importance 用的是未标准化值 * 系数（系数来自标准化后的拟合）
    # 这会影响绝对值，但不影响“按 sport 排序”的主目的。
    imp = sport_importance(res_total, res_gold, x_total, x_gold, pred_feat[["NOC"] + [f"int_total__{s}" for s in top_sports] + [f"int_gold__{s}" for s in top_sports]], top_sports)
    imp.to_csv(os.path.join(OUT_DIR, "sport_importance_top5_by_country.csv"), index=False)

    # 主场效应（美国 2028）：host=1 vs host=0 的差值（点预测）
    usa_row = pred_df[pred_df["NOC"] == "USA"].copy()
    if len(usa_row) > 0:
        # 简单输出（更完整可做反事实预测：把 pred_feat USA 的 is_host 置 0 重新走一次 predict）
        pass

    print("Done. Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
