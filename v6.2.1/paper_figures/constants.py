"""Paper-wide visual identity for methods and stages."""

METHOD_STYLES = {
    "Ours": {"color": "#1769AA", "marker": "o"},
    "Minimum-Jerk": {"color": "#D55E00", "marker": "s"},
    "RRT-Connect": {"color": "#009E73", "marker": "^"},
    "STOMP": {"color": "#CC79A7", "marker": "D"},
    "cuRobo": {"color": "#7A5195", "marker": "P"},
    "Invalid": {"color": "#B7B7B7", "marker": "."},
}

STAGE_COLORS = {
    "Grasp": "#1769AA",
    "Place": "#E69F00",
    "Joint": "#009E73",
    "Failure": "#C44E52",
}

SINGLE_COLUMN_IN = 3.35
DOUBLE_COLUMN_IN = 7.0
RNG_SEED = 20260814
