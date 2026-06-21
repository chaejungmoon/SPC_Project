import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(
    page_title="공정능력분석 & SPC 웹앱",
    layout="wide"
)

# =========================
# 강의록 기반 상수표
# =========================
CONST = {
    2:  {"A2": 1.880, "A3": 2.659, "d2": 1.128, "D3": 0.000, "D4": 3.267, "B3": 0.000, "B4": 3.267, "c4": 0.7979},
    3:  {"A2": 1.023, "A3": 1.954, "d2": 1.693, "D3": 0.000, "D4": 2.574, "B3": 0.000, "B4": 2.568, "c4": 0.8862},
    4:  {"A2": 0.729, "A3": 1.628, "d2": 2.059, "D3": 0.000, "D4": 2.282, "B3": 0.000, "B4": 2.266, "c4": 0.9213},
    5:  {"A2": 0.577, "A3": 1.427, "d2": 2.326, "D3": 0.000, "D4": 2.114, "B3": 0.000, "B4": 2.089, "c4": 0.9400},
    6:  {"A2": 0.483, "A3": 1.287, "d2": 2.534, "D3": 0.000, "D4": 2.004, "B3": 0.030, "B4": 1.970, "c4": 0.9515},
    7:  {"A2": 0.419, "A3": 1.182, "d2": 2.704, "D3": 0.076, "D4": 1.924, "B3": 0.118, "B4": 1.882, "c4": 0.9594},
    8:  {"A2": 0.373, "A3": 1.099, "d2": 2.847, "D3": 0.136, "D4": 1.864, "B3": 0.185, "B4": 1.815, "c4": 0.9650},
    9:  {"A2": 0.337, "A3": 1.032, "d2": 2.970, "D3": 0.184, "D4": 1.816, "B3": 0.239, "B4": 1.761, "c4": 0.9693},
    10: {"A2": 0.308, "A3": 0.975, "d2": 3.078, "D3": 0.223, "D4": 1.777, "B3": 0.284, "B4": 1.716, "c4": 0.9727},
}


def get_const(n, name):
    n = int(n)
    if n in CONST:
        return CONST[n][name]
    return CONST[10][name]


def grade_cpk(cpk):
    if cpk >= 1.67:
        return "매우 충분"
    elif cpk >= 1.33:
        return "충분"
    elif cpk >= 1.00:
        return "보통 / 개선 권장"
    elif cpk >= 0.67:
        return "부족"
    else:
        return "매우 부족"


def make_sample_variable():
    np.random.seed(42)
    df = pd.DataFrame({
        "subgroup": np.repeat(np.arange(1, 26), 5),
        "value": np.random.normal(100, 4, 125)
    })
    df.loc[df["subgroup"] == 8, "value"] += 12
    df.loc[df["subgroup"] == 19, "value"] -= 10
    return df


def make_sample_attribute():
    np.random.seed(42)
    return pd.DataFrame({
        "lot": np.arange(1, 26),
        "sample_size": np.repeat(200, 25),
        "defects": np.random.binomial(200, 0.03, 25)
    })


def variable_summary(df, subgroup_col, value_col):
    data = df[[subgroup_col, value_col]].dropna().copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()

    g = data.groupby(subgroup_col)[value_col]

    summary = pd.DataFrame({
        "mean": g.mean(),
        "range": g.max() - g.min(),
        "std": g.std(ddof=1),
        "n": g.count()
    })

    n = int(summary["n"].mode()[0])
    return data, summary, n


def capability(data, summary, value_col, lsl, usl):
    values = data[value_col]
    mean = values.mean()
    n = int(summary["n"].mode()[0])

    rbar = summary["range"].mean()
    sbar = summary["std"].mean()

    sigma_within_r = rbar / get_const(n, "d2")
    sigma_within_s = sbar / get_const(n, "c4")
    sigma_within = sigma_within_s

    sigma_overall = values.std(ddof=1)

    cp = (usl - lsl) / (6 * sigma_within)
    cpk = min(
        (usl - mean) / (3 * sigma_within),
        (mean - lsl) / (3 * sigma_within)
    )

    pp = (usl - lsl) / (6 * sigma_overall)
    ppk = min(
        (usl - mean) / (3 * sigma_overall),
        (mean - lsl) / (3 * sigma_overall)
    )

    return {
        "mean": mean,
        "sigma_within": sigma_within,
        "sigma_overall": sigma_overall,
        "Cp": cp,
        "Cpk": cpk,
        "Pp": pp,
        "Ppk": ppk,
        "grade": grade_cpk(cpk)
    }


def control_chart(df, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["point"],
        mode="lines+markers",
        name="관측값"
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["CL"], mode="lines", name="CL"))
    fig.add_trace(go.Scatter(x=df.index, y=df["UCL"], mode="lines", name="UCL"))
    fig.add_trace(go.Scatter(x=df.index, y=df["LCL"], mode="lines", name="LCL"))

    out = df[(df["point"] > df["UCL"]) | (df["point"] < df["LCL"])]

    if len(out) > 0:
        fig.add_trace(go.Scatter(
            x=out.index,
            y=out["point"],
            mode="markers",
            marker=dict(size=13, color="red"),
            name="이상점"
        ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        xaxis_title="Index",
        yaxis_title="Value"
    )

    return fig, out


def xbar_r(summary, n):
    xbarbar = summary["mean"].mean()
    rbar = summary["range"].mean()

    A2 = get_const(n, "A2")
    D3 = get_const(n, "D3")
    D4 = get_const(n, "D4")

    xchart = pd.DataFrame({
        "point": summary["mean"],
        "CL": xbarbar,
        "UCL": xbarbar + A2 * rbar,
        "LCL": xbarbar - A2 * rbar
    })

    rchart = pd.DataFrame({
        "point": summary["range"],
        "CL": rbar,
        "UCL": D4 * rbar,
        "LCL": D3 * rbar
    })

    return xchart, rchart


def xbar_s(summary, n):
    xbarbar = summary["mean"].mean()
    sbar = summary["std"].mean()

    A3 = get_const(n, "A3")
    B3 = get_const(n, "B3")
    B4 = get_const(n, "B4")

    xchart = pd.DataFrame({
        "point": summary["mean"],
        "CL": xbarbar,
        "UCL": xbarbar + A3 * sbar,
        "LCL": xbarbar - A3 * sbar
    })

    schart = pd.DataFrame({
        "point": summary["std"],
        "CL": sbar,
        "UCL": B4 * sbar,
        "LCL": B3 * sbar
    })

    return xchart, schart


def imr(data, subgroup_col, value_col):
    x = data.sort_values(subgroup_col)[value_col].reset_index(drop=True)
    mr = x.diff().abs()

    xbar = x.mean()
    mrbar = mr.dropna().mean()

    d2 = 1.128
    D3 = 0
    D4 = 3.267

    ichart = pd.DataFrame({
        "point": x,
        "CL": xbar,
        "UCL": xbar + 3 * mrbar / d2,
        "LCL": xbar - 3 * mrbar / d2
    })

    mrchart = pd.DataFrame({
        "point": mr,
        "CL": mrbar,
        "UCL": D4 * mrbar,
        "LCL": D3 * mrbar
    })

    return ichart, mrchart


def p_np(df, lot_col, sample_col, defect_col):
    data = df[[lot_col, sample_col, defect_col]].dropna().copy()
    data[sample_col] = pd.to_numeric(data[sample_col], errors="coerce")
    data[defect_col] = pd.to_numeric(data[defect_col], errors="coerce")
    data = data.dropna()

    pbar = data[defect_col].sum() / data[sample_col].sum()

    pchart = pd.DataFrame(index=data[lot_col])
    pchart["point"] = data[defect_col].values / data[sample_col].values
    pchart["CL"] = pbar
    pchart["UCL"] = pbar + 3 * np.sqrt(pbar * (1 - pbar) / data[sample_col].values)
    pchart["LCL"] = pbar - 3 * np.sqrt(pbar * (1 - pbar) / data[sample_col].values)
    pchart["LCL"] = pchart["LCL"].clip(lower=0)

    nbar = data[sample_col].mean()
    npbar = nbar * pbar

    npchart = pd.DataFrame(index=data[lot_col])
    npchart["point"] = data[defect_col].values
    npchart["CL"] = npbar
    npchart["UCL"] = npbar + 3 * np.sqrt(nbar * pbar * (1 - pbar))
    npchart["LCL"] = npbar - 3 * np.sqrt(nbar * pbar * (1 - pbar))
    npchart["LCL"] = npchart["LCL"].clip(lower=0)

    return pchart, npchart, pbar


def hist_with_normal(values, lsl, usl):
    mean = values.mean()
    std = values.std(ddof=1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        histnorm="probability density",
        name="데이터 분포"
    ))

    x = np.linspace(values.min(), values.max(), 200)
    y = norm.pdf(x, mean, std)

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="정규분포"
    ))

    fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")

    fig.update_layout(
        title="히스토그램 + 정규분포 + 규격한계",
        template="plotly_white",
        height=430
    )
    return fig

def box_plot(values):

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=values,
            name="공정 데이터",
            boxpoints="outliers"
        )
    )

    fig.update_layout(
        title="Box Plot",
        template="plotly_white",
        height=400
    )

    return fig


# =========================
# 웹앱 UI
# =========================
st.title("📊 공정능력분석 + 통계적공정관리 SPC 웹앱")
st.write("강의록의 공정능력분석과 통계적공정관리를 기반으로 데이터 업로드, 지수 계산, 관리도 작성, 이상판정, 개선 전후 비교를 수행합니다.")

st.sidebar.header("분석 설정")

data_kind = st.sidebar.radio(
    "데이터 유형 선택",
    ["계량형 데이터", "계수형 데이터"]
)

uploaded = st.sidebar.file_uploader("CSV 업로드", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    if data_kind == "계량형 데이터":
        df = make_sample_variable()
    else:
        df = make_sample_attribute()

st.subheader("데이터 미리보기")
st.dataframe(df.head(20), use_container_width=True)

tabs = st.tabs(["공정능력분석", "관리도", "이상판정/개선", "자동 리포트"])

if data_kind == "계량형 데이터":
    cols = df.columns.tolist()

    subgroup_col = st.sidebar.selectbox("부분군 열 선택", cols, index=0)
    value_col = st.sidebar.selectbox("측정값 열 선택", cols, index=1)

    lsl = st.sidebar.number_input("LSL", value=80.0)
    usl = st.sidebar.number_input("USL", value=120.0)
    target = st.sidebar.number_input("Target", value=100.0)

    data, summary, n = variable_summary(df, subgroup_col, value_col)
    cap = capability(data, summary, value_col, lsl, usl)

    with tabs[0]:
        st.header("공정능력분석")

        x_r_temp, _ = xbar_r(summary, n)

        abnormal_count = len(
            x_r_temp[
                (x_r_temp["point"] > x_r_temp["UCL"]) |
                (x_r_temp["point"] < x_r_temp["LCL"])])

        st.subheader("공정 상태 Dashboard")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cp", f"{cap['Cp']:.3f}")
        c2.metric("Cpk", f"{cap['Cpk']:.3f}")
        c3.metric("Ppk", f"{cap['Ppk']:.3f}")
        c4.metric("이상점 수", abnormal_count)

        c1, c2, c3 = st.columns(3)
        c1.metric("평균", f"{cap['mean']:.3f}")
        c2.metric("σ within", f"{cap['sigma_within']:.3f}")
        c3.metric("σ overall", f"{cap['sigma_overall']:.3f}")

        st.success(f"공정능력 판정: {cap['grade']}")

        fig = hist_with_normal(data[value_col], lsl, usl)
        st.plotly_chart(fig, use_container_width=True)

        st.plotly_chart(
            box_plot(data[value_col]),
            use_container_width=True)

    with tabs[1]:
        st.header("관리도")

        x_r, r = xbar_r(summary, n)
        x_s, s = xbar_s(summary, n)
        i_chart, mr_chart = imr(data, subgroup_col, value_col)

        st.subheader("Xbar-R 관리도")
        fig, out_xr = control_chart(x_r, "Xbar Chart")
        st.plotly_chart(fig, use_container_width=True)
        fig, out_r = control_chart(r, "R Chart")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Xbar-S 관리도")
        fig, out_xs = control_chart(x_s, "Xbar Chart")
        st.plotly_chart(fig, use_container_width=True)
        fig, out_s = control_chart(s, "S Chart")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("I-MR 관리도")
        fig, out_i = control_chart(i_chart, "I Chart")
        st.plotly_chart(fig, use_container_width=True)
        fig, out_mr = control_chart(mr_chart, "MR Chart")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.header("이상판정 및 개선 전후 비교")

        x_r, r = xbar_r(summary, n)
        _, out_xr = control_chart(x_r, "Xbar Chart")

        abnormal_groups = list(out_xr.index)

        if len(abnormal_groups) == 0:
            st.success("Xbar 관리도 기준 이상 부분군이 없습니다.")
        else:
            st.error(f"이상 부분군: {abnormal_groups}")

            improved_data = data[~data[subgroup_col].isin(abnormal_groups)]
            imp_data, imp_summary, imp_n = variable_summary(improved_data, subgroup_col, value_col)
            imp_cap = capability(imp_data, imp_summary, value_col, lsl, usl)

            compare = pd.DataFrame({
                "구분": ["개선 전", "개선 후"],
                "Cp": [cap["Cp"], imp_cap["Cp"]],
                "Cpk": [cap["Cpk"], imp_cap["Cpk"]],
                "Pp": [cap["Pp"], imp_cap["Pp"]],
                "Ppk": [cap["Ppk"], imp_cap["Ppk"]],
                "평균": [cap["mean"], imp_cap["mean"]],
                "σwithin": [cap["sigma_within"], imp_cap["sigma_within"]],
                "σoverall": [cap["sigma_overall"], imp_cap["sigma_overall"]],
            })

            st.dataframe(compare, use_container_width=True)

            imp_xr, imp_r = xbar_r(imp_summary, imp_n)
            fig, _ = control_chart(imp_xr, "개선 후 Xbar Chart")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.header("자동 리포트")

        st.markdown(f"""
        ### 분석 결과 요약

        - 분석 데이터: **계량형 데이터**
        - 부분군 열: **{subgroup_col}**
        - 측정값 열: **{value_col}**
        - 부분군 크기: **{n}**
        - LSL: **{lsl}**
        - USL: **{usl}**
        - Target: **{target}**
        
        ### 공정능력지수
        - Cp = **{cap['Cp']:.3f}**
        - Cpk = **{cap['Cpk']:.3f}**
        - Pp = **{cap['Pp']:.3f}**
        - Ppk = **{cap['Ppk']:.3f}**

        ### 해석

        현재 공정의 Cpk는 **{cap['Cpk']:.3f}**이며, 판정 결과는 **{cap['grade']}**입니다.

        Cp/Cpk는 군내변동을 이용한 단기 공정능력이고,
        Pp/Ppk는 전체변동을 이용한 장기 공정성능입니다.

        ### 개선 제안

        """)

        if cap['Cpk'] >= 1.33:
            st.success("공정능력이 충분한 상태입니다. 현재 조건을 유지하며 지속적으로 모니터링하는 것을 권장합니다.")
        elif cap['Cpk'] >= 1.00:
            st.warning("공정능력은 확보되어 있으나 개선 여지가 있습니다. 산포 감소 및 중심값 조정을 검토하세요.")
        else:
            st.error("공정능력이 부족합니다. 이상원인 분석 및 공정 개선이 필요합니다.")

        report_text = f"""
        SPC REPORT

        Cp : {cap['Cp']:.3f}
        Cpk : {cap['Cpk']:.3f}
        Pp : {cap['Pp']:.3f}
        Ppk : {cap['Ppk']:.3f}

        평균 : {cap['mean']:.3f}
        σ within : {cap['sigma_within']:.3f}
        σ overall : {cap['sigma_overall']:.3f}

        판정 : {cap['grade']}
        """

        st.download_button(
        label="📄 보고서 다운로드",
        data=report_text,
        file_name="SPC_Report.txt",
        mime="text/plain"
        )

else:
    cols = df.columns.tolist()

    lot_col = st.sidebar.selectbox("Lot 열 선택", cols, index=0)
    sample_col = st.sidebar.selectbox("검사수 열 선택", cols, index=1)
    defect_col = st.sidebar.selectbox("불량수 열 선택", cols, index=2)

    pchart, npchart, pbar = p_np(df, lot_col, sample_col, defect_col)

    with tabs[0]:
        st.header("계수형 데이터 요약")
        st.metric("전체 불량률 p-bar", f"{pbar:.4f}")

    with tabs[1]:
        st.header("P / NP 관리도")

        fig, out_p = control_chart(pchart, "P Chart")
        st.plotly_chart(fig, use_container_width=True)

        fig, out_np = control_chart(npchart, "NP Chart")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.header("이상판정")

        _, out_p = control_chart(pchart, "P Chart")
        _, out_np = control_chart(npchart, "NP Chart")

        st.subheader("P 관리도 이상점")
        st.dataframe(out_p, use_container_width=True)

        st.subheader("NP 관리도 이상점")
        st.dataframe(out_np, use_container_width=True)

    with tabs[3]:
        st.header("자동 리포트")

        st.markdown(f"""
        ### 분석 결과 요약

        - 분석 데이터: **계수형 데이터**
        - Lot 열: **{lot_col}**
        - 검사수 열: **{sample_col}**
        - 불량수 열: **{defect_col}**
        - 전체 불량률 p-bar: **{pbar:.4f}**

        ### 해석

        P 관리도는 로트별 불량률을 관리하고, NP 관리도는 로트별 불량수를 관리합니다.  
        관리한계를 벗어난 로트는 이상 원인 검토가 필요합니다.
        """)