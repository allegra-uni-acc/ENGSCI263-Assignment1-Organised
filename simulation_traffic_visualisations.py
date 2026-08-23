
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Force interactive GUI window on macOS/PyCharm
plt.switch_backend("TkAgg")

# ==============================================================================
# CONFIGURATION & REPO FILE PATHS
# ==============================================================================
DEMAND_FILE = Path("project_data/FoodstuffsDemand2026.csv")
ROUTES_FILE = Path("project_data/feasible_routes.csv")

N_BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 263
DAYS_PER_WEEK = 7

# Exact LP Weekly Baseline Pallet Demand
BASELINE_LP_WEEKLY_PALLETS = 674.0

# Fleet Capacity Limit (Adjust if your daily truck/shift pallet limit differs)
# 20 trucks * 2 shifts * 16 pallets/truck = 640 pallets/day -> 4,480/wk
# Set here to match your fleet's daily capacity scaled to weekly
DAILY_FLEET_PALLET_CAPACITY = 20 * 2 * 16
WEEKLY_FLEET_PALLET_CAPACITY = DAILY_FLEET_PALLET_CAPACITY * DAYS_PER_WEEK


# ==============================================================================
# DEMAND GRAPH FUNCTION (PALLETS)
# ==============================================================================
def plot_weekly_pallet_demand(weekly_sim_df: pd.DataFrame, baseline_lp_pallets: float = BASELINE_LP_WEEKLY_PALLETS):
    pallets = weekly_sim_df["weekly_total_pallets"]
    p95_pallets = pallets.quantile(0.95)
    mean_pallets = pallets.mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    # 1. Plot simulated weekly pallet demand distribution
    ax.hist(
        pallets,
        bins=35,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.8,
        label="Simulated Weekly Pallet Demand (1,000 Runs)",
    )

    # 2. Linear Program Baseline Line (674 Pallets)
    ax.axvline(
        baseline_lp_pallets,
        color="crimson",
        linewidth=2.5,
        linestyle="--",
        label=f"MIP Baseline LP Demand ({baseline_lp_pallets:,.0f} Pallets/Wk)",
    )

    # 3. 95th Percentile Demand Target Line
    ax.axvline(
        p95_pallets,
        color="darkorange",
        linewidth=2.5,
        linestyle=":",
        label=f"95% CI Simulated Demand ({p95_pallets:,.0f} Pallets/Wk)",
    )

    ax.set_title(
        "MIP Optimal Baseline vs. Realized Weekly Stochastic Pallet Demand",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Total Weekly Pallet Demand across 60 Optimal Routes", fontsize=11)
    ax.set_ylabel("Frequency (Simulated Weeks)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Callout Annotation
    pallet_gap = p95_pallets - baseline_lp_pallets
    ax.annotate(
        f"Realized Demand Overhead:\nMean: {mean_pallets:,.0f} Pallets/Wk\n95% CI: {p95_pallets:,.0f} Pallets/Wk\n"
        f"Exceeds LP Baseline by {pallet_gap:,.0f} Pallets/Wk\ndue to stochastic store demand surges.",
        xy=(p95_pallets, 15),
        xytext=(p95_pallets + (p95_pallets * 0.03), 35),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        fontsize=9,
    )

    ax.legend(loc="upper left")
    plt.tight_layout()

    # Save to disk first and then open window
    plt.savefig("weekly_pallet_demand_vs_lp_baseline.png", dpi=300)
    plt.show(block=True)


# ==============================================================================
# MAIN SIMULATION ENGINE
# ==============================================================================
def run_simulation():
    rng = np.random.default_rng(RANDOM_SEED)

    if not DEMAND_FILE.exists() or not ROUTES_FILE.exists():
        raise FileNotFoundError("Check that FoodstuffsDemand2026.csv and your routes CSV exist.")

    demand_df = pd.read_csv(DEMAND_FILE)
    routes_df = pd.read_csv(ROUTES_FILE)

    # Filter to MIP optimal routes
    if "is_selected" in routes_df.columns:
        optimal_routes = routes_df[routes_df["is_selected"] == 1].copy()
    elif "Selected" in routes_df.columns:
        optimal_routes = routes_df[routes_df["Selected"] == 1].copy()
    else:
        optimal_routes = routes_df.head(60).copy()

    n_routes = len(optimal_routes)
    n_days = len(demand_df)

    numeric_demand_df = demand_df.select_dtypes(include=[np.number])
    overall_mean_demand = numeric_demand_df.values.mean()
    daily_demand_scales = (numeric_demand_df.mean(axis=1) / overall_mean_demand).to_numpy()

    # Scale base route pallets so that mean weekly base aligns with your LP baseline
    raw_pallets_base = (
        optimal_routes["Pallets"].to_numpy(dtype=float)
        if "Pallets" in optimal_routes
        else np.ones(n_routes)
    )

    # Scale factor ensures baseline week matches your 674 pallets
    daily_target_pallets = BASELINE_LP_WEEKLY_PALLETS / DAYS_PER_WEEK
    scale_to_lp = daily_target_pallets / raw_pallets_base.sum()
    pallets_base = raw_pallets_base * scale_to_lp

    weekly_results = []

    # Run 1,000 simulated WEEKS
    for sim_idx in range(1, N_BOOTSTRAP_SAMPLES + 1):
        week_day_indices = rng.choice(n_days, size=DAYS_PER_WEEK, replace=True)
        weekly_pallets = 0.0

        for day_idx in week_day_indices:
            demand_scale = daily_demand_scales[day_idx]

            # Realized daily pallet demand across optimal routes
            actual_daily_pallets = (pallets_base * demand_scale).sum()
            weekly_pallets += actual_daily_pallets

        weekly_results.append({
            "simulation_week": sim_idx,
            "weekly_total_pallets": weekly_pallets
        })

    weekly_sim_df = pd.DataFrame(weekly_results)

    p95_pallets = weekly_sim_df["weekly_total_pallets"].quantile(0.95)

    print("\n==============================================")
    print("WEEKLY PALLET DEMAND SIMULATION RESULTS")
    print("==============================================")
    print(f"Optimal Routes Evaluated:         {n_routes}")
    print(f"MIP LP Baseline Weekly Pallets:  {BASELINE_LP_WEEKLY_PALLETS:,.0f} Pallets")
    print(f"Mean Weekly Simulated Pallets:   {weekly_sim_df['weekly_total_pallets'].mean():,.0f} Pallets")
    print(f"95th Percentile Weekly Demand:    {p95_pallets:,.0f} Pallets")

    # Plot graph
    plot_weekly_pallet_demand(weekly_sim_df, baseline_lp_pallets=BASELINE_LP_WEEKLY_PALLETS)


if __name__ == "__main__":
    run_simulation()

Traffic_simulation_demand_graph.py

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Force interactive GUI window on macOS/PyCharm
plt.switch_backend("TkAgg")

# ==============================================================================
# CONFIGURATION & REPO FILE PATHS
# ==============================================================================
DEMAND_FILE = Path("FoodstuffsDemand2026.csv")
ROUTES_FILE = Path("All_Feasible_Routes_Simplified.csv")

N_BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 263
DAYS_PER_WEEK = 7  # Scaling daily simulations to weekly horizons

# Logistics Constants
UNLOAD_MINUTES_PER_PALLET = 18.0
TRUCK_COST_PER_HOUR = 220.0
OVERTIME_THRESHOLD_HOURS = 4.0
OVERTIME_COST_PER_BLOCK = 310.0
WET_LEASE_COST_PER_2H_BLOCK = 1400.0

# Fleet Limit: 20 trucks * 2 shifts * 3.5 hours = 140 available fleet hours per day
NORMAL_TRUCK_HOURS_DAILY = 20 * 2 * 3.5

# Traffic Multipliers
DAY_TO_DAY_SIGMA = 0.30
ROUTE_TO_ROUTE_SIGMA = 0.15


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def ceil_hours_over_threshold(total_hours: float, threshold: float) -> int:
    extra = total_hours - threshold
    if extra <= 0:
        return 0
    return int(math.ceil(extra - 1e-12))


# ==============================================================================
# DEMAND & CONFIDENCE GRAPH FUNCTION (WEEKLY SCALE)
# ==============================================================================
def plot_weekly_cost_comparison(weekly_sim_df: pd.DataFrame, baseline_lp_weekly_cost: float = 275203.11):
    weekly_costs = weekly_sim_df["weekly_total_cost"]
    p95_cost = weekly_costs.quantile(0.95)
    mean_cost = weekly_costs.mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    # 1. Plot the 1,000 simulated weekly runs
    ax.hist(
        weekly_costs,
        bins=35,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.8,
        label="Simulated Actual Weekly Cost (1,000 Runs)",
    )

    # 2. Linear Program Weekly Baseline Line ($275,203.11)
    ax.axvline(
        baseline_lp_weekly_cost,
        color="crimson",
        linewidth=2.5,
        linestyle="--",
        label=f"Weekly MIP Baseline Cost (${baseline_lp_weekly_cost:,.2f})",
    )

    # 3. 95th Percentile Weekly Cost Line
    ax.axvline(
        p95_cost,
        color="darkorange",
        linewidth=2.5,
        linestyle=":",
        label=f"95% CI Simulated Weekly Cost (${p95_cost:,.2f})",
    )

    ax.set_title(
        "MIP Optimal Schedule Baseline vs. Realized Weekly Stochastic Costs",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Total Weekly Operational Cost ($)", fontsize=11)
    ax.set_ylabel("Frequency (Simulated Weeks)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Management Callout Annotation
    cost_gap = p95_cost - baseline_lp_weekly_cost
    ax.annotate(
        f"Realized Weekly Overhead:\nMean: ${mean_cost:,.0f}\n95% CI: ${p95_cost:,.0f}\n"
        f"Exceeds LP Baseline by ${cost_gap:,.0f}\ndue to traffic & wet-leasing.",
        xy=(p95_cost, 15),
        xytext=(p95_cost + (p95_cost * 0.03), 35),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        fontsize=9,
    )

    ax.legend(loc="upper left")
    plt.tight_layout()

    # Save to disk first
    plt.savefig("lp_baseline_vs_weekly_simulation_cost.png", dpi=300)

    # Display plot
    plt.show(block=True)


# ==============================================================================
# MAIN SIMULATION ENGINE
# ==============================================================================
def run_simulation():
    rng = np.random.default_rng(RANDOM_SEED)

    if not DEMAND_FILE.exists() or not ROUTES_FILE.exists():
        raise FileNotFoundError("Check that FoodstuffsDemand2026.csv and your routes CSV exist.")

    demand_df = pd.read_csv(DEMAND_FILE)
    routes_df = pd.read_csv(ROUTES_FILE)

    # Filter to MIP optimal routes
    if "is_selected" in routes_df.columns:
        optimal_routes = routes_df[routes_df["is_selected"] == 1].copy()
    elif "Selected" in routes_df.columns:
        optimal_routes = routes_df[routes_df["Selected"] == 1].copy()
    else:
        optimal_routes = routes_df.head(60).copy()

    n_routes = len(optimal_routes)
    n_days = len(demand_df)

    numeric_demand_df = demand_df.select_dtypes(include=[np.number])
    overall_mean_demand = numeric_demand_df.values.mean()
    daily_demand_scales = (numeric_demand_df.mean(axis=1) / overall_mean_demand).to_numpy()

    print(f"Running simulation on {n_routes} MIP optimal routes scaled to weekly totals...")

    pallets_base = (
        optimal_routes["Pallets"].to_numpy(dtype=float)
        if "Pallets" in optimal_routes
        else np.ones(n_routes)
    )
    unload_hours_base = (pallets_base * UNLOAD_MINUTES_PER_PALLET) / 60.0

    base_drive_hours = np.maximum(
        0.0,
        optimal_routes["Duration_hours"].to_numpy(dtype=float) - unload_hours_base
    )

    weekly_results = []

    # Run 1,000 simulated WEEKS (7 sampled operational days per week)
    for sim_idx in range(1, N_BOOTSTRAP_SAMPLES + 1):
        # Sample 7 random days from historical dataset for this week
        week_day_indices = rng.choice(n_days, size=DAYS_PER_WEEK, replace=True)

        weekly_cost = 0.0
        weekly_hours = 0.0
        weekly_wet_blocks = 0

        for day_idx in week_day_indices:
            demand_scale = daily_demand_scales[day_idx]

            day_mult = rng.lognormal(mean=-0.5 * DAY_TO_DAY_SIGMA ** 2, sigma=DAY_TO_DAY_SIGMA)
            route_mult = rng.lognormal(mean=-0.5 * ROUTE_TO_ROUTE_SIGMA ** 2, sigma=ROUTE_TO_ROUTE_SIGMA, size=n_routes)

            actual_unload_hours = unload_hours_base * demand_scale
            actual_route_hours = (base_drive_hours * day_mult * route_mult) + actual_unload_hours
            total_daily_hours = actual_route_hours.sum()

            normal_truck_cost = total_daily_hours * TRUCK_COST_PER_HOUR
            overtime_blocks = sum(ceil_hours_over_threshold(h, OVERTIME_THRESHOLD_HOURS) for h in actual_route_hours)
            overtime_cost = overtime_blocks * OVERTIME_COST_PER_BLOCK

            extra_hours = max(0.0, total_daily_hours - NORMAL_TRUCK_HOURS_DAILY)
            wet_lease_blocks = int(math.ceil(extra_hours / 2.0 - 1e-12)) if extra_hours > 0 else 0
            wet_lease_cost = wet_lease_blocks * WET_LEASE_COST_PER_2H_BLOCK

            daily_total_cost = normal_truck_cost + overtime_cost + wet_lease_cost

            # Accumulate into weekly metrics
            weekly_cost += daily_total_cost
            weekly_hours += total_daily_hours
            weekly_wet_blocks += wet_lease_blocks

        weekly_results.append({
            "simulation_week": sim_idx,
            "weekly_total_hours": weekly_hours,
            "weekly_wet_lease_blocks": weekly_wet_blocks,
            "weekly_total_cost": weekly_cost
        })

    weekly_sim_df = pd.DataFrame(weekly_results)

    p95_weekly_cost = weekly_sim_df["weekly_total_cost"].quantile(0.95)

    print("\n==============================================")
    print("WEEKLY SIMULATION & BOOTSTRAP RESULTS")
    print("==============================================")
    print(f"Optimal Routes Evaluated:         {n_routes}")
    print(f"Mean Weekly Total Cost:          ${weekly_sim_df['weekly_total_cost'].mean():,.2f}")
    print(f"95th Percentile Weekly Cost:     ${p95_weekly_cost:,.2f}")
    print(f"LP Baseline Weekly Cost:         $275,203.11")
    print(f"Mean Total Required Weekly Hrs:  {weekly_sim_df['weekly_total_hours'].mean():.2f} hrs")

    # Plot weekly comparison
    plot_weekly_cost_comparison(weekly_sim_df, baseline_lp_weekly_cost=275203.11)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    run_simulation()