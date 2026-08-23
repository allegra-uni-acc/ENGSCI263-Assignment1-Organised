from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# USER SETTINGS
# =========================

ROUTES_FILE = Path("Optimal_Routes.csv")
OUTPUT_DIR = Path("figures")

OVERTIME_THRESHOLD_HOURS = 3.5
WET_LEASE_THRESHOLD_HOURS = 4.0

DAY_COLORS = {
    "Weekdays": "#2E86AB",
    "Saturday": "#E67E22",
}


def load_routes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Num_Stops"] = df["Route"].apply(
        lambda r: sum(1 for s in r.split("->") if s.strip().lower() != "warehouse")
    )
    return df


def _color_for(day: str) -> str:
    return DAY_COLORS.get(day, "#888888")


def plot_cost_per_route(df: pd.DataFrame, ax=None):
    """Bar chart of trip cost per route, sorted descending, coloured by day."""
    ax = ax or plt.gca()
    sorted_df = df.sort_values("Trip_Cost_NZD", ascending=False).reset_index(drop=True)
    colors = [_color_for(d) for d in sorted_df["Day"]]

    ax.bar(range(len(sorted_df)), sorted_df["Trip_Cost_NZD"], color=colors)
    ax.set_xlabel("Route (sorted by cost)")
    ax.set_ylabel("Trip cost (NZD)")
    ax.set_title("Trip Cost per Route")
    _add_day_legend(ax)


def plot_duration_per_route(df: pd.DataFrame, ax=None):
    """Bar chart of route duration, with reference lines at 3.5h / 4h thresholds."""
    ax = ax or plt.gca()
    sorted_df = df.sort_values("Duration_hours", ascending=False).reset_index(drop=True)
    colors = [_color_for(d) for d in sorted_df["Day"]]

    ax.bar(range(len(sorted_df)), sorted_df["Duration_hours"], color=colors)
    ax.axhline(OVERTIME_THRESHOLD_HOURS, color="black", linestyle="--", linewidth=1,
               label=f"{OVERTIME_THRESHOLD_HOURS}h target")
    ax.axhline(WET_LEASE_THRESHOLD_HOURS, color="red", linestyle="--", linewidth=1,
               label=f"{WET_LEASE_THRESHOLD_HOURS}h overtime rate kicks in")
    ax.set_xlabel("Route (sorted by duration)")
    ax.set_ylabel("Duration (hours)")
    ax.set_title("Route Duration")
    ax.legend(fontsize=8)


def plot_cost_vs_duration(df: pd.DataFrame, ax=None):
    """Scatter of cost vs duration, coloured by day, sized by demand (pallets)."""
    ax = ax or plt.gca()
    for day, g in df.groupby("Day"):
        ax.scatter(g["Duration_hours"], g["Trip_Cost_NZD"],
                   s=g["Demand"] * 8, alpha=0.6,
                   color=_color_for(day), label=day, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Trip cost (NZD)")
    ax.set_title("Cost vs Duration (bubble size = pallets)")
    ax.legend()


def plot_total_cost_by_day(df: pd.DataFrame, ax=None):
    """Total daily fleet cost, split by day."""
    ax = ax or plt.gca()
    totals = df.groupby("Day")["Trip_Cost_NZD"].sum()
    colors = [_color_for(d) for d in totals.index]
    bars = ax.bar(totals.index, totals.values, color=colors)
    ax.set_ylabel("Total cost (NZD)")
    ax.set_title("Total Fleet Cost by Day")
    for bar, val in zip(bars, totals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"${val:,.0f}",
                ha="center", va="bottom", fontsize=9)


def plot_route_count_and_hours_by_day(df: pd.DataFrame, ax=None):
    """Number of routes and total truck-hours used, per day, against the 140h cap."""
    ax = ax or plt.gca()
    hours = df.groupby("Day")["Duration_hours"].sum()
    colors = [_color_for(d) for d in hours.index]
    bars = ax.bar(hours.index, hours.values, color=colors)
    ax.axhline(140, color="black", linestyle="--", linewidth=1,
               label="Owned fleet capacity (140h = 20 trucks x 2 shifts x 3.5h)")
    ax.set_ylabel("Total truck-hours")
    ax.set_title("Truck-Hours Used vs Owned Fleet Capacity")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, hours.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.1f}h",
                ha="center", va="bottom", fontsize=9)


def plot_demand_distribution(df: pd.DataFrame, ax=None):
    """Histogram of pallets delivered per route."""
    ax = ax or plt.gca()
    for day, g in df.groupby("Day"):
        ax.hist(g["Demand"], bins=range(0, 20, 2), alpha=0.6,
                color=_color_for(day), label=day, edgecolor="white")
    ax.set_xlabel("Pallets per route")
    ax.set_ylabel("Number of routes")
    ax.set_title("Demand (Pallets) per Route")
    ax.legend()


def _add_day_legend(ax):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DAY_COLORS.values()]
    ax.legend(handles, DAY_COLORS.keys(), fontsize=8)


def build_dashboard(df: pd.DataFrame, save_path: Path | None = None):
    """Combine all six charts into a single dashboard figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Foodstuffs Trucking Schedule -- Cost & Duration Overview", fontsize=15, y=0.995)

    plot_cost_per_route(df, axes[0, 0])
    plot_duration_per_route(df, axes[0, 1])
    plot_cost_vs_duration(df, axes[1, 0])
    plot_total_cost_by_day(df, axes[1, 1])
    plot_route_count_and_hours_by_day(df, axes[2, 0])
    plot_demand_distribution(df, axes[2, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved dashboard to {save_path}")

    return fig


def main() -> None:
    df = load_routes(ROUTES_FILE)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Individual charts, saved separately
    chart_functions = {
        "cost_per_route": plot_cost_per_route,
        "duration_per_route": plot_duration_per_route,
        "cost_vs_duration": plot_cost_vs_duration,
        "total_cost_by_day": plot_total_cost_by_day,
        "hours_vs_capacity": plot_route_count_and_hours_by_day,
        "demand_distribution": plot_demand_distribution,
    }

    for name, func in chart_functions.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        func(df, ax)
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    # Combined dashboard
    build_dashboard(df, save_path=OUTPUT_DIR / "dashboard.png")


if __name__ == "__main__":
    main()