import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Use browser renderer
pio.renderers.default = "browser"

# 1) Read the merged CSV
df = pd.read_csv("merged_nudat_data.csv")

# 2) Ensure 'a' exists
if "a" not in df.columns:
    df["a"] = df["z"] + df["n"]

# 3) Parse half-life into numeric seconds, leave NaN for "STABLE" or missing
def parse_halflife(val):
    try:
        return float(val)
    except:
        return np.nan

df["halflife_sec"] = df["halflife"].apply(parse_halflife)
df["log_halflife"] = np.log10(df["halflife_sec"])

# 4) Split into three subsets
df_stable = df[df["halflife"] == "STABLE"]
df_unknown = df[df["halflife"].isna()]
df_radio  = df[(df["halflife"].notna()) & (df["halflife"] != "STABLE")]

# 5) Build the figure and add each trace

fig = go.Figure()

# 5a) Stable nuclides → black squares
fig.add_trace(
    go.Scatter(
        x=df_stable["n"],
        y=df_stable["z"],
        mode="markers",
        marker=dict(
            color="black",
            symbol="square",
            size=6
        ),
        showlegend=False,
        name="Stable"
    )
)

# 5b) Unknown nuclides → light gray circles
fig.add_trace(
    go.Scatter(
        x=df_unknown["n"],
        y=df_unknown["z"],
        mode="markers",
        marker=dict(
            color="lightgray",
            symbol="square",
            size=6
        ),
        showlegend=False,
        name="Unknown"
    )
)

# 5c) Radioactive nuclides → colored by log_halflife with Plasma scale
#    We explicitly set cmin / cmax so the color scale spans the data range.
cmin = df_radio["log_halflife"].min()
cmax = df_radio["log_halflife"].max()

fig.add_trace(
    go.Scatter(
        x=df_radio["n"],
        y=df_radio["z"],
        mode="markers",
        marker=dict(
            color=df_radio["log_halflife"],
            colorscale="Viridis",
            colorbar=dict(title="log₁₀(τ₁/₂ [s])"),
            symbol="square",  # CHANGED from default
            size=6,
            cmin=cmin,
            cmax=cmax
        ),
        showlegend=False,
        name="Radioactive"
    )
)

# 6) Configure axes and layout
fig.update_layout(
    title="Chart of Nuclides (N vs Z)",
    xaxis=dict(
        title="Neutron Number (N)",
        range=[0, 180],      # NEW RANGE
        autorange=False,     # Force range to stick
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title="Proton Number (Z)",
        range=[0, 120],
        autorange=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        title="Nuclear Status",
        itemsizing="constant"
    ),
    width=1250,
    height=900
)



# 7) Add hovertemplate to show useful info when you hover
#    We'll include A, Z, N, half-life (if any), and a couple of other columns like B2 or Sn as an example.
hover_template = (
    "A = %{customdata[0]}<br>" +
    "Z = %{y}<br>" +
    "N = %{x}<br>" +
    "Half-life = %{customdata[1]}<br>" +
    "β<sub>2</sub> = %{customdata[2]}<br>" +
    "S<sub>n</sub> = %{customdata[3]}<extra></extra>"
)

# Attach customdata arrays for each trace
# Stable: no half-life needed
fig.data[0].update(
    customdata=np.stack([
        df_stable["a"],
        df_stable["halflife"],
        df_stable["B2"],
        df_stable["Sn"]
    ], axis=-1),
    hovertemplate=hover_template
)

# Unknown: no numeric half-life (it’s NaN)
fig.data[1].update(
    customdata=np.stack([
        df_unknown["a"],
        df_unknown["halflife"],  # likely NaN
        df_unknown["B2"],
        df_unknown["Sn"]
    ], axis=-1),
    hovertemplate=hover_template
)

# Radioactive: include the numeric half-life
fig.data[2].update(
    customdata=np.stack([
        df_radio["a"],
        df_radio["halflife"],
        df_radio["B2"],
        df_radio["Sn"]
    ], axis=-1),
    hovertemplate=hover_template
)

# 8) Show the figure in the browser
fig.show()

