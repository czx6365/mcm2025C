# -*- coding: utf-8 -*-
"""
Bullet 1: 奖牌数预测 + 预测区间（Prediction Interval）
输出：outputs/la2028_predictions.csv
"""

import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from config import DATA_RAW_DIR, OUTPUT_DIR

# ========================
# 基本参数
# ========================
ATH = DATA_RAW_DIR / "summerOly_athletes.csv"
HOST = DATA_RAW_DIR / "summerOly_hosts.csv"
PROG = DATA_RAW_DIR / "summerOly_programs.csv"
MEDALS = DATA_RAW_DIR / "summerOly_medal_counts.csv"

OUT_DIR = OUTPUT_DIR
OUT_DIR.mkdir(exist_ok=True)

TRAIN_END = 2024
PRED_YEAR = 2028
N_SIM = 5000
PI_LEVELS = [(0.10, 0.90), (0.025, 0.975)]
np.random.seed(42)

# ========================
# 工具函数
# ========================

def clean_host_country(x):
    """把 'Paris, France' -> 'France'"""
    if pd.isna(x):
        return ""
    return str(x).replace("\xa0", " ").split(",")[-1].strip()


def nb_mu_to_params(mu, alpha):
    """NB(mu, alpha) -> numpy NB(n,p)"""
    alpha = max(float(alpha), 1e-12)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return n, p


def prepare_design(df, x_cols, scaler=None, fit=True):
    """标准化特征；预测阶段 fit=False 使用同一 scaler"""
    X = df[x_cols].astype(float).values
    if scaler is None:
        scaler = StandardScaler()
    Xs = scaler.fit_transform(X) if fit else scaler.transform(X)
    return pd.DataFrame(Xs, columns=x_cols, index=df.index), scaler


def drop_constant_and_duplicate_cols(X: pd.DataFrame) -> pd.DataFrame:
    """
    删除常数列、重复列（完全相同），减少奇异矩阵风险
    """
    # 常数列
    nun = X.nunique(dropna=False)
    keep = nun[nun > 1].index.tolist()
    X = X[keep].copy()

    # 重复列（完全一致）
    X = X.loc[:, ~X.T.duplicated()]
    return X


def fit_nb_stable(df, y_col, x_cols):
    """
    更稳健的 NB 拟合：
    - 自动去掉常数/重复列
    - 若 Hessian 奇异，换优化器，必要时加极小 ridge（只影响数值稳定，不改变模型结构）
    """
    X = df[x_cols].copy()
    X = drop_constant_and_duplicate_cols(X)
    X = sm.add_constant(X, has_constant="add")
    y = df[y_col].astype(float).values

    # 先用默认拟合
    try:
        res = sm.NegativeBinomial(y, X).fit(disp=False, maxiter=300)
        return res
    except Exception:
        pass

    # 换优化器
    try:
        res = sm.NegativeBinomial(y, X).fit(disp=False, maxiter=300, method="bfgs")
        return res
    except Exception:
        pass

    # 最后：极小 ridge（仅用于数值稳定）
    # 注：statsmodels 的 NB 没有直接 ridge 选项，这里通过轻微扰动 X 来避免完全共线
    X_r = X.copy()
    eps = 1e-8
    X_r.iloc[:, 1:] = X_r.iloc[:, 1:] + eps * np.random.randn(*X_r.iloc[:, 1:].shape)
    res = sm.NegativeBinomial(y, X_r).fit(disp=False, maxiter=300, method="bfgs")
    return res


# ========================
# 正确的奖牌与规模定义
# ========================

def build_medal_targets(medal_counts: pd.DataFrame) -> pd.DataFrame:
    """
    奖牌口径：来自 medal_counts.csv（官方口径）
    """
    df = medal_counts.copy()
    df = df.rename(columns={"Gold": "gold_medals", "Total": "total_medals"})
    df = df[["NOC", "Year", "gold_medals", "total_medals"]].copy()
    df["gold_medals"] = pd.to_numeric(df["gold_medals"], errors="coerce").fillna(0.0)
    df["total_medals"] = pd.to_numeric(df["total_medals"], errors="coerce").fillna(0.0)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["NOC"] = df["NOC"].astype(str).str.strip()
    return df


def build_country_scale_features(athletes: pd.DataFrame) -> pd.DataFrame:
    """
    athletes.csv 只用于规模/结构特征（不数奖牌！）
    """
    df = athletes.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["NOC"] = df["NOC"].astype(str).str.strip()
    return (
        df.groupby(["NOC", "Year"], as_index=False)
          .agg(
              athletes_cnt=("Name", "nunique"),
              sports_cnt=("Sport", "nunique"),
          )
    )


def build_host_map(hosts: pd.DataFrame, athletes: pd.DataFrame) -> dict:
    """
    hosts 的 Host 字段是国家/城市名，需要映射成 NOC。
    用 athletes 的 Team->NOC 众数做映射。
    """
    a = athletes.copy()
    a["Team"] = a["Team"].astype(str).str.strip()
    a["NOC"] = a["NOC"].astype(str).str.strip()

    team2noc = (
        a.groupby("Team")["NOC"]
         .agg(lambda s: s.value_counts().index[0])
         .to_dict()
    )

    m = {}
    for _, r in hosts.iterrows():
        y = int(r["Year"])
        country = clean_host_country(r["Host"])
        m[y] = team2noc.get(country, "")
    return m


def maybe_map_medals_noc(medals: pd.DataFrame, athletes: pd.DataFrame) -> pd.DataFrame:
    """
    medal_counts 的 NOC 是国家名（如 United States / Great Britain），
    需要映射成 athletes 的三字母 NOC（USA/GBR/...）
    做法：
    1) 用 athletes 的 Team->NOC 众数映射
    2) 加一组常见别名/历史名手工修正（很重要）
    3) 映射后只保留合法三字母 NOC，否则丢弃（避免把模型喂坏）
    """
    df = medals.copy()
    df["NOC_raw"] = df["NOC"].astype(str).str.strip()
    df["NOC"] = df["NOC_raw"]

    # athletes: Team -> NOC（众数）
    a = athletes.copy()
    a["Team"] = a["Team"].astype(str).str.strip()
    a["NOC"] = a["NOC"].astype(str).str.strip()
    team2noc = (
        a.groupby("Team")["NOC"]
         .agg(lambda s: s.value_counts().index[0])
         .to_dict()
    )

    # 常见别名/历史名修正（按你数据集的写法可能还要补几条）
    alias = {
        "United States": "USA",
        "Great Britain": "GBR",
        "Britain": "GBR",
        "Russia": "RUS",
        "Russian Federation": "RUS",
        "ROC": "RUS",
        "Soviet Union": "URS",
        "East Germany": "GDR",
        "West Germany": "FRG",
        "Germany": "GER",
        "Czech Republic": "CZE",
        "Czechoslovakia": "TCH",
        "Yugoslavia": "YUG",
        "Serbia and Montenegro": "SCG",
        "United Arab Republic": "UAR",
        "China": "CHN",
        "Hong Kong": "HKG",
        "Chinese Taipei": "TPE",
        "Formosa": "TPE",
        "Korea, South": "KOR",
        "South Korea": "KOR",
        "Korea, North": "PRK",
        "North Korea": "PRK",
        "Iran": "IRI",
        "Egypt": "EGY",
        "Türkiye": "TUR",
        "Turkey": "TUR",
    }

    # 先用 alias，再用 team2noc 回退
    df["NOC_mapped"] = df["NOC"].map(alias)
    df["NOC_mapped"] = df["NOC_mapped"].fillna(df["NOC"].map(team2noc))
    df["NOC"] = df["NOC_mapped"].fillna(df["NOC"])
    df = df.drop(columns=["NOC_mapped"])

    # 只保留三字母 NOC（关键：否则 merge 后 athletes_cnt=0 会污染训练）
    is_code = df["NOC"].astype(str).str.fullmatch(r"[A-Z]{3}").fillna(False)
    keep_df = df[is_code].copy()

    # (NOC,Year) 去重聚合
    keep_df = keep_df.sort_values(["NOC", "Year", "total_medals"])
    keep_df = keep_df.drop_duplicates(["NOC", "Year"], keep="last")

    dup2 = keep_df.groupby(["NOC", "Year"]).size().max()
    assert dup2 == 1, "Found duplicated (NOC,Year) after cleaning medals!"

    # 覆盖率提示（可留着，论文里也好解释数据清洗）
    coverage = len(keep_df) / max(len(df), 1)
    if coverage < 0.95:
        print(f"[WARN] medal NOC mapping coverage: {coverage:.2%} (<95%). "
              f"Unmapped rows dropped to protect model stability.")

    return keep_df



def build_events(programs: pd.DataFrame) -> pd.DataFrame:
    """
    从 programs 表聚合得到每年总赛事数，并取 log 控制供给侧规模
    """
    year_cols = [c for c in programs.columns if re.fullmatch(r"\d{4}\*?", str(c))]
    long = programs.melt("Sport", year_cols, "Year", "events")
    long["Year"] = long["Year"].astype(str).str.replace("*", "", regex=False).astype(int)
    long["events"] = pd.to_numeric(long["events"], errors="coerce").fillna(0.0)

    ev = long.groupby("Year", as_index=False)["events"].sum()
    ev = ev.rename(columns={"events": "events_total_all_sports"})
    ev["log_events"] = np.log(ev["events_total_all_sports"] + 1.0)
    return ev[["Year", "events_total_all_sports", "log_events"]]


# ========================
# 构造建模表
# ========================

def build_table(athletes, medal_counts, hosts, programs):
    # 1) 国家规模（参赛国家-年份全集）
    scale = build_country_scale_features(athletes)

    # 2) 奖牌（正确口径 + 映射成三字母 NOC）
    medals = build_medal_targets(medal_counts)
    medals = maybe_map_medals_noc(medals, athletes)

    # ✅ 关键：以 scale 为底表，奖牌左合并；没拿牌就 0
    table = scale.merge(medals, on=["NOC", "Year"], how="left")
    table["gold_medals"] = table["gold_medals"].fillna(0.0)
    table["total_medals"] = table["total_medals"].fillna(0.0)

    # 3) host
    host_map = build_host_map(hosts, athletes)
    table["host_noc"] = table["Year"].map(host_map).fillna("")
    table["host_boost"] = (table["NOC"] == table["host_noc"]).astype(float)
    table["host_boost"] *= 0.1


    # 4) events
    ev = build_events(programs)
    table = table.merge(ev[["Year", "log_events"]], on="Year", how="left").fillna(0.0)

    # 5) lag & EMA（国家惯性）
    table = table.sort_values(["NOC", "Year"]).reset_index(drop=True)
    for c in ["total_medals", "gold_medals", "athletes_cnt", "sports_cnt"]:
        table[f"lag1_{c}"] = table.groupby("NOC")[c].shift(1).fillna(0.0)
        table[f"ema_{c}"] = (
            table.groupby("NOC")[c]
                 .shift(1)
                 .ewm(span=3, adjust=False)
                 .mean()
                 .fillna(0.0)
        )

    # ??????? EMA??? exp ?????
    table["log1p_ema_total_medals"] = np.log1p(table["ema_total_medals"])
    table["log1p_ema_gold_medals"] = np.log1p(table["ema_gold_medals"])

    # 6) log1p 规模
    for c in ["athletes_cnt", "lag1_athletes_cnt", "sports_cnt", "lag1_sports_cnt"]:
        table[f"log1p_{c}"] = np.log1p(table[c])

    # 7) 年份清洗
    table = table[(table["Year"] <= TRAIN_END) & (table["Year"] != 1906)].copy()
    return table



# ========================
# 不确定性模拟（PI）
# ========================

def simulate_pi(res, Xp, n_sim=N_SIM):
    """
    - beta 参数不确定性：多元正态抽样
    - NB 计数不确定性：negative_binomial 抽样
    """
    try:
        alpha = float(np.exp(res.params["lnalpha"]))
    except Exception:
        alpha = float(res.params.get("alpha", 0.2))

    beta_names = [c for c in res.params.index if c in Xp.columns]
    beta_mean = res.params.loc[beta_names].values

    # 协方差可能不可得/不正定，做保护
    try:
        cov = res.cov_params().loc[beta_names, beta_names].values
        beta_draw = np.random.multivariate_normal(beta_mean, cov, size=n_sim)
    except Exception:
        beta_draw = np.tile(beta_mean, (n_sim, 1))

    eta = beta_draw @ Xp[beta_names].values.T
    mu_draw = np.exp(eta)

    n, p = nb_mu_to_params(mu_draw, alpha)
    y_sim = np.random.negative_binomial(n=n, p=p).astype(float)

    out = {"mu_point": res.predict(Xp).values}
    for lo, hi in PI_LEVELS:
        out[(lo, hi)] = (
            np.quantile(y_sim, lo, axis=0),
            np.quantile(y_sim, hi, axis=0),
        )
    return out


# ========================
# 主流程
# ========================

def pick_baseline_year(table: pd.DataFrame) -> int:
    """
    预测基准年：优先 2024，其次 2020，否则取数据中最大年份
    """
    years = sorted(table["Year"].unique())
    if 2024 in years:
        return 2024
    if 2020 in years:
        return 2020
    return years[-1]


def main():
    athletes = pd.read_csv(ATH)
    hosts = pd.read_csv(HOST)
    programs = pd.read_csv(PROG, encoding="latin1")
    medal_counts = pd.read_csv(MEDALS)

    table = build_table(athletes, medal_counts, hosts, programs)

    # 两套特征：避免共线、且 Gold 模型要用 gold 惯性
    x_cols_total = [
        "host_boost",
        "log_events",
        "log1p_athletes_cnt",
        "log1p_lag1_athletes_cnt",
        "log1p_ema_total_medals",
    ]
    x_cols_gold = [
        "host_boost",
        "log_events",
        "log1p_athletes_cnt",
        "log1p_lag1_athletes_cnt",
        "log1p_ema_gold_medals",
    ]

    # ===== 训练：Total =====
    Xs_t, scaler_t = prepare_design(table, x_cols_total, fit=True)
    train_t = table.copy()
    train_t[x_cols_total] = Xs_t
    res_total = fit_nb_stable(train_t, "total_medals", x_cols_total)

    # ===== 训练：Gold =====
    Xs_g, scaler_g = prepare_design(table, x_cols_gold, fit=True)
    train_g = table.copy()
    train_g[x_cols_gold] = Xs_g
    res_gold = fit_nb_stable(train_g, "gold_medals", x_cols_gold)

    # ===== 构造预测集（基准年状态 -> 2028）=====
    base_year = pick_baseline_year(table)
    latest = table[table["Year"] == base_year].copy()

    # 只预测“近年活跃”的 NOC：用 athletes 的最近两届（若不存在则用所有）
    yrs_avail = sorted(pd.to_numeric(athletes["Year"], errors="coerce").dropna().astype(int).unique())
    cand = [y for y in [2024, 2020] if y in yrs_avail]
    if len(cand) == 0:
        active_nocs = athletes["NOC"].astype(str).unique()
    else:
        active_nocs = athletes[athletes["Year"].isin(cand)]["NOC"].astype(str).unique()
    latest = latest[latest["NOC"].astype(str).isin(active_nocs)].copy()

    # 设定预测年份与主办国
    latest["Year"] = PRED_YEAR
    # 显式设定 2028 的 log_events：默认用基准年近似（论文写：use latest known as proxy）
    logev_base = table.loc[table["Year"] == base_year, "log_events"].iloc[0]
    latest["log_events"] = float(logev_base)

    # ===== 预测：Total =====
    Xp_t_std, _ = prepare_design(latest, x_cols_total, scaler=scaler_t, fit=False)
    Xp_t = sm.add_constant(Xp_t_std, has_constant="add")
    sim_total = simulate_pi(res_total, Xp_t)

    # ===== 预测：Gold =====
    Xp_g_std, _ = prepare_design(latest, x_cols_gold, scaler=scaler_g, fit=False)
    Xp_g = sm.add_constant(Xp_g_std, has_constant="add")
    sim_gold = simulate_pi(res_gold, Xp_g)

    # ===== 输出 =====
    out = pd.DataFrame({
        "NOC": latest["NOC"].values,
        "Year": PRED_YEAR,
        "total_pred": sim_total["mu_point"],
        "gold_pred": sim_gold["mu_point"],
    })

    for lo, hi in PI_LEVELS:
        out[f"total_{int(lo*100)}_{int(hi*100)}_lo"] = sim_total[(lo, hi)][0]
        out[f"total_{int(lo*100)}_{int(hi*100)}_hi"] = sim_total[(lo, hi)][1]
        out[f"gold_{int(lo*100)}_{int(hi*100)}_lo"] = sim_gold[(lo, hi)][0]
        out[f"gold_{int(lo*100)}_{int(hi*100)}_hi"] = sim_gold[(lo, hi)][1]

    out = out.sort_values(["gold_pred", "total_pred"], ascending=False).reset_index(drop=True)

    out_path = OUT_DIR / "la2028_predictions.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(f"Baseline year used: {base_year} -> Pred year: {PRED_YEAR}")
    print("Top 10 preview:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
