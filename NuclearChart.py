import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Use browser renderer
pio.renderers.default = "browser"

# 1) Load data
df = pd.read_csv("merged_nudat_data.csv")

# 2) Add A if needed
if "a" not in df.columns:
    df["a"] = df["z"] + df["n"]

# 3) Parse half-life
def parse_halflife(val):
    try:
        return float(val)
    except:
        return np.nan

df["halflife_sec"] = df["halflife"].apply(parse_halflife)
df["log_halflife"] = np.log10(df["halflife_sec"])

# 4) Separate subsets
df_stable = df[df["halflife"] == "STABLE"]
df_unknown = df[df["halflife"].isna()]
df_radio = df[(df["halflife"].notna()) & (df["halflife"] != "STABLE")]

# 5) Create pivoted 2D array for Heatmap (radioactive only)
heatmap_data = df_radio.pivot(index="z", columns="n", values="log_halflife")

# 6) Create figure
fig = go.Figure()

# Heatmap for radioactive nuclei
fig.add_trace(go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale="Viridis",
    zmin=df_radio["log_halflife"].min(),
    zmax=df_radio["log_halflife"].max(),
    colorbar=dict(title="log₁₀(τ₁/₂ [s])"),
    hovertemplate=(
        "Z = %{y}<br>" +
        "N = %{x}<br>" +
        "log₁₀(Half-life [s]) = %{z:.2f}<extra></extra>"
    )
))

# Add stable nuclei as black squares
fig.add_trace(go.Scatter(
    x=df_stable["n"],
    y=df_stable["z"],
    mode="markers",
    marker=dict(color="black", symbol="square", size=6),
    showlegend=False
))

# Add unknown nuclei as light gray
fig.add_trace(go.Scatter(
    x=df_unknown["n"],
    y=df_unknown["z"],
    mode="markers",
    marker=dict(color="lightgray", symbol="square", size=6),
    showlegend=False
))

# Layout
fig.update_layout(
    title="Chart of Nuclides (Heatmap + Scatter)",
    xaxis=dict(
        title="Neutron Number (N)",
        range=[0, 180],
        showgrid=False,
        zeroline=False,
        autorange=False
    ),
    yaxis=dict(
        title="Proton Number (Z)",
        range=[0, 120],
        showgrid=False,
        zeroline=False,
        autorange=False,
        scaleanchor="x",
        scaleratio=1
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1250,
    height=900
)

fig.show()

