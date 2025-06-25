import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as pcolors

pio.renderers.default = "browser"

# 1) Read the merged CSV
df = pd.read_csv("merged_nudat_data.csv")

# 2) Ensure 'a' exists
if "a" not in df.columns:
    df["a"] = df["z"] + df["n"]

# 3) Parse half-life into numeric seconds (NaN if “STABLE” or missing)
def parse_halflife(val):
    try:
        return float(val)
    except:
        return np.nan

df["halflife_sec"] = df["halflife"].apply(parse_halflife)
df["log_halflife"] = np.log10(df["halflife_sec"])

# 4) Determine min/max for color‐mapping radioactive cells
radio_mask = df["halflife"].notna() & (df["halflife"] != "STABLE")
if radio_mask.any():
    rmin = df.loc[radio_mask, "log_halflife"].min()
    rmax = df.loc[radio_mask, "log_halflife"].max()
else:
    rmin = rmax = 0.0

# 5) Prepare a Viridis palette to sample colors
viridis = pcolors.sequential.Viridis  # a list of hex‐colors, length ≈256
n_colors = len(viridis)

def get_viridis_color(val):
    """Normalize val between rmin and rmax, then pick a Viridis hex."""
    if np.isnan(val):
        return None
    t = (val - rmin) / (rmax - rmin) if (rmax > rmin) else 0.0
    idx = int(np.clip(t * (n_colors - 1), 0, n_colors - 1))
    return viridis[idx]

# 6) Build a list of rectangle‐shapes for every nucleus
shapes = []

for _, row in df.iterrows():
    z = int(row["z"])
    n = int(row["n"])
    # Each rectangle spans [n-0.5, n+0.5] × [z-0.5, z+0.5]
    x0, x1 = n - 0.49, n + 0.49
    y0, y1 = z - 0.49, z + 0.49

    if row["halflife"] == "STABLE":
        fill = "black"
    elif pd.isna(row["halflife"]):
        fill = "lightgray"
    else:
        fill = get_viridis_color(row["log_halflife"])

    # Skip if somehow fill is None (shouldn't happen here)
    if fill is None:
        continue

    shapes.append(
        dict(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=fill,
            line_width=0,
        )
    )

# 7) Build an invisible scatter trace purely for hover‐tooltips
#    We place one marker at each (n, z) with opacity=0. Each marker carries customdata.
hover_scatter = go.Scatter(
    x=df["n"],
    y=df["z"],
    mode="markers",
    marker=dict(color="rgba(0,0,0,0)", size=1),
    customdata=np.stack([df["a"], df["halflife"], df["B2"], df["Sn"]], axis=-1),
    hovertemplate=(
        "A = %{customdata[0]}<br>"
        "Z = %{y}<br>"
        "N = %{x}<br>"
        "Half-life = %{customdata[1]}<br>"
        "B<sub>2</sub> = %{customdata[2]}<br>"
        "S<sub>n</sub> = %{customdata[3]}<extra></extra>"
    ),
    showlegend=False
)

# 8) Create the figure, add shapes, then overlay the hover‐scatter
fig = go.Figure()

fig.update_layout(
    shapes=shapes,
    title="Chart of Nuclides",
    xaxis=dict(
        title="Neutron Number (N)",
        range=[-0.5, 180.5],
        autorange=False,
        showgrid=False,
        zeroline=False,
    ),
    yaxis=dict(
        title="Proton Number (Z)",
        range=[-0.5, 120.5],
        autorange=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1250,
    height=900,
)

fig.add_trace(hover_scatter)
fig.show()
