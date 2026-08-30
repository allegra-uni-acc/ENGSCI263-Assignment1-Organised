from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# =========================

# IMPORTANT TO READ HERE:
# This new file should in theory model the traffic in such a way that accounts for what kind of shift it uses.
# The general way that it works is by reading the Scheduled optimal routes rather than all routes and then judging based off of that
# rather than by reading all possible paths.
# This should in theory allow for the numbers to look significantly better than they did before

ROUTE_FILE = Path(__file__).resolve().parent / "project_data" / "Scheduled_Optimal_Routes.csv"

N_SIMULATIONS = 1000
RANDOM_SEED = 263

# --- Traffic variation ---------------------------------------------------
# Shared across all routes in the same (Day, Shift) group in a given
# simulation -- represents "today's 8am peak was worse/better than usual".
SHIFT_TO_SHIFT_SIGMA = 0.25
# Independent variation for each individual route on top of that.
ROUTE_TO_ROUTE_SIGMA = 0.15

# ASSUMPTION: mean traffic multiplier per shift. 8am overlaps the morning
# commute peak more heavily than 2pm (which only tails into the evening
# peak toward the end of a long route); both are centred so a "typical"
# day reproduces roughly the durations already in the schedule.
# ADJUST THESE if you guys find real traffic-index data to calibrate against.
SHIFT_TRAFFIC_MEAN = {
    "8am": 1.10,
    "2pm": 1.05,
}

# --- Demand variation ------------------------------------------------------
# Day-to-day variation in pallets ordered per store, as a fraction of the
# planned (baseline) demand on that route. May need to adjust based off of noise within the data set but I think this is
# fine
DEMAND_SIGMA_FRACTION = 0.15

# --- Project constants -------------------------------------------------
UNLOAD_MINUTES_PER_PALLET = 18.0
TRUCK_COST_PER_HOUR = 220.0
OVERTIME_THRESHOLD_HOURS = 4.0
OVERTIME_COST_PER_HOUR = 310.0
WET_LEASE_COST_PER_2H_BLOCK = 1400.0

TRUCKS_PER_SHIFT = 20
SHIFT_TARGET_HOURS = 3.5


def parse_route(route_text: str) -> List[str]:
    return [p.strip() for p in str(route_text).split("->") if p.strip()]


def route_cost(hours: float) -> float:
    """
    Cost of a single route given its (simulated) total hours.
    Hours up to OVERTIME_THRESHOLD_HOURS are billed at TRUCK_COST_PER_HOUR;
    hours beyond that are billed at OVERTIME_COST_PER_HOUR INSTEAD of the
    normal rate for those extra hours (not stacked on top of it), per the
    brief: "the extra time costs Foodstuffs $310 per hour (or part thereof)".
    """
    if hours <= OVERTIME_THRESHOLD_HOURS:
        return hours * TRUCK_COST_PER_HOUR
    normal_part = OVERTIME_THRESHOLD_HOURS * TRUCK_COST_PER_HOUR
    overtime_hours = math.ceil(hours - OVERTIME_THRESHOLD_HOURS - 1e-12)  # charged per hour or part thereof
    return normal_part + overtime_hours * OVERTIME_COST_PER_HOUR


def simulate_group(group: pd.DataFrame, n_simulations: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Run the Monte Carlo simulation for one (Day) worth of routes -- e.g. all
    'Weekdays' routes, or all 'Saturday' routes -- across both shifts.

    Capacity is modelled the way the MIP actually models it: a fixed number
    of trucks per shift (not an aggregate hour budget), so an individual
    route running long just costs overtime -- it doesn't need a wet-lease
    substitute UNLESS that truck is also booked for the OTHER shift that day
    and the overrun pushes its finish time past the next shift's start. That
    truck-level conflict is the genuine capacity risk traffic can create
    here, so that's what's simulated as the wet-lease trigger.
    """
    group = group.reset_index(drop=True)
    baseline_pallets = group["Demand"].to_numpy(dtype=float)
    baseline_unload_hours = baseline_pallets * UNLOAD_MINUTES_PER_PALLET / 60.0
    baseline_drive_hours = np.maximum(0.0, group["Duration_hours"].to_numpy() - baseline_unload_hours)
    shifts = group["Scheduled_Shift"].to_numpy()
    trucks = group["Truck"].to_numpy()
    n_routes = len(group)

    # Identify trucks double-booked across both shifts on this day, and the
    # row positions of their AM / PM routes.
    truck_shift_rows = {}
    for i, (truck, shift) in enumerate(zip(trucks, shifts)):
        truck_shift_rows.setdefault(truck, {})[shift] = i
    double_shift_trucks = {
        truck: rows for truck, rows in truck_shift_rows.items()
        if "8am" in rows and "2pm" in rows
    }
    shift_start_hour = {"8am": 8.0, "2pm": 14.0}

    rows = []
    for sim in range(1, n_simulations + 1):

        # ---- demand variability (per route, per simulation) ----
        demand_sim = rng.normal(baseline_pallets, baseline_pallets * DEMAND_SIGMA_FRACTION)
        demand_sim = np.clip(np.round(demand_sim), 0, None)
        unload_hours_sim = demand_sim * UNLOAD_MINUTES_PER_PALLET / 60.0

        # ---- traffic variability (shift-level + route-level) ----
        traffic_factor = np.empty(n_routes)
        for shift_name, mean_mult in SHIFT_TRAFFIC_MEAN.items():
            mask = shifts == shift_name
            if not mask.any():
                continue
            shift_draw = rng.lognormal(mean=math.log(mean_mult) - 0.5 * SHIFT_TO_SHIFT_SIGMA ** 2,
                                        sigma=SHIFT_TO_SHIFT_SIGMA)
            route_draws = rng.lognormal(mean=-0.5 * ROUTE_TO_ROUTE_SIGMA ** 2,
                                         sigma=ROUTE_TO_ROUTE_SIGMA, size=mask.sum())
            traffic_factor[mask] = shift_draw * route_draws

        actual_hours = baseline_drive_hours * traffic_factor + unload_hours_sim

        # ---- per-route cost, assuming every route runs as owned ----
        costs = np.array([route_cost(h) for h in actual_hours])

        # ---- truck-level conflict check: does an AM overrun eat into this
        #      truck's own PM slot? If so, that PM route needs a wet-lease
        #      substitute instead of the owned truck. ----
        wet_lease_cost = 0.0
        n_conflicts = 0
        for truck, rows_by_shift in double_shift_trucks.items():
            am_idx = rows_by_shift["8am"]
            pm_idx = rows_by_shift["2pm"]
            am_finish_clock = shift_start_hour["8am"] + actual_hours[am_idx]
            if am_finish_clock > shift_start_hour["2pm"]:
                n_conflicts += 1
                # swap that PM route's cost from owned-rate to wet-lease-rate
                costs[pm_idx] = math.ceil(actual_hours[pm_idx] / 2.0 - 1e-12) * WET_LEASE_COST_PER_2H_BLOCK
                wet_lease_cost += costs[pm_idx]

        total_cost = costs.sum()

        rows.append({
            "simulation": sim,
            "total_hours": actual_hours.sum(),
            "routes_over_3_5h": int((actual_hours > 3.5).sum()),
            "routes_over_4h": int((actual_hours > 4.0).sum()),
            "shift_conflicts": n_conflicts,
            "wet_lease_cost": wet_lease_cost,
            "total_cost": total_cost,
        })

    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, day_label: str, n_routes: int) -> pd.DataFrame:
    return pd.DataFrame({
        "metric": [
            "Routes scheduled", "Mean total cost", "Median total cost",
            "95th percentile total cost", "Mean total truck-hours",
            "Mean # routes > 3.5h", "P(any route > 4h)",
            "P(a shift-conflict occurs)", "Mean # conflicts per sim",
            "Mean wet lease cost",
        ],
        f"{day_label}": [
            n_routes,
            results["total_cost"].mean(),
            results["total_cost"].median(),
            results["total_cost"].quantile(0.95),
            results["total_hours"].mean(),
            results["routes_over_3_5h"].mean(),
            (results["routes_over_4h"] > 0).mean(),
            (results["shift_conflicts"] > 0).mean(),
            results["shift_conflicts"].mean(),
            results["wet_lease_cost"].mean(),
        ],
    })


def main() -> None:
    if not ROUTE_FILE.exists():
        raise FileNotFoundError(f"Missing {ROUTE_FILE.resolve()}")

    routes = pd.read_csv(ROUTE_FILE)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Loaded {len(routes):,} scheduled routes across "
          f"{routes['Day'].nunique()} day-type(s): {list(routes['Day'].unique())}\n")

    summaries = []
    for day_label, group in routes.groupby("Day"):
        print(f"Simulating {day_label} ({len(group)} routes, "
              f"{N_SIMULATIONS:,} simulations)...")
        results = simulate_group(group, N_SIMULATIONS, rng)
        summaries.append(summarise(results, day_label, len(group)))

    combined = summaries[0]
    for s in summaries[1:]:
        combined = combined.merge(s, on="metric")

    print("\n==============================")
    print("SIMULATION RESULTS (by day-type)")
    print("==============================")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()