from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# =========================

ROUTE_FILE = Path(__file__).resolve().parent / "project_data" / "Scheduled_Optimal_Routes.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "project_data" / "simulation_traffic_results.csv"

N_SIMULATIONS = 1000
RANDOM_SEED = 263

# --- Traffic variation ---------------------------------------------------
# Shared across all routes in the same (Day, Shift) group in a given
# simulation -- represents "today's 8am peak was worse/better than usual".
SHIFT_TO_SHIFT_SIGMA = 0.10
# Independent variation for each individual route on top of that.
ROUTE_TO_ROUTE_SIGMA = 0.06

# ASSUMPTION: mean traffic multiplier per shift. 8am overlaps the morning
# commute peak more heavily than 2pm (which only tails into the evening
# peak toward the end of a long route); both are centred so a "typical"
# day reproduces roughly the durations already in the schedule.
# ADJUST THESE if you have real traffic-index data to calibrate against.
SHIFT_TRAFFIC_MEAN = {
    "8am": 1.03,
    "2pm": 1.02,
}

# --- Project constants -------------------------------------------------
UNLOAD_MINUTES_PER_PALLET = 18.0
TRUCK_COST_PER_HOUR = 220.0
OVERTIME_THRESHOLD_HOURS = 4.0
OVERTIME_COST_PER_HOUR = 310.0
WET_LEASE_COST_PER_2H_BLOCK = 1400.0

# SIMPLIFICATION: a truck whose AM route finishes only slightly after 14:00
# is treated as NOT needing a wet-lease substitute for its PM route -- e.g.
# the driver making up a few minutes, or the 2pm departure having a little
# real-world flex. Only overruns PAST this grace period count as a genuine
# conflict. ADJUST or set to 0 to go back to a strict "any overrun = conflict"
# model.
CONFLICT_GRACE_PERIOD_MINUTES = 15.0


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

    TRAFFIC ONLY: pallet demand is held at its baseline (planned) value
    every simulation -- demand is modelled separately elsewhere, not here.
    Only driving time varies, via a traffic multiplier. Because demand never
    changes from the planned figure (which was already capacity-feasible
    when the route was built), there's no capacity-overflow risk to model
    here -- that's a demand-driven risk, out of scope for this file.

    Capacity is modelled the way the MIP actually models it: a fixed number
    of trucks per shift (not an aggregate hour budget), so an individual
    route running long just costs overtime -- it doesn't need a wet-lease
    substitute UNLESS that truck is also booked for the OTHER shift that day
    and the overrun pushes its finish time past the next shift's start. That
    truck-level conflict is the genuine capacity risk traffic creates here.
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

        # demand held at its planned baseline every simulation -- unload
        # time is fixed, not simulated
        actual_hours = baseline_drive_hours * traffic_factor + baseline_unload_hours

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
            grace_hours = CONFLICT_GRACE_PERIOD_MINUTES / 60.0
            if am_finish_clock > shift_start_hour["2pm"] + grace_hours:
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


def main() -> None:
    if not ROUTE_FILE.exists():
        raise FileNotFoundError(f"Missing {ROUTE_FILE.resolve()}")

    routes = pd.read_csv(ROUTE_FILE)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Loaded {len(routes):,} scheduled routes across "
          f"{routes['Day'].nunique()} day-type(s): {list(routes['Day'].unique())}\n")

    all_results = []
    for day_label, group in routes.groupby("Day"):
        print(f"Simulating traffic for {day_label} ({len(group)} routes, "
              f"{N_SIMULATIONS:,} simulations)...")
        results = simulate_group(group, N_SIMULATIONS, rng)
        results.insert(0, "Day", day_label)
        all_results.append(results)

    combined_results = pd.concat(all_results, ignore_index=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_results.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print("TRAFFIC-ONLY SIMULATION SUMMARY")
    print("==============================")
    summary = combined_results.groupby("Day").agg(
        Mean_Total_Cost=("total_cost", "mean"),
        Median_Total_Cost=("total_cost", "median"),
        P95_Total_Cost=("total_cost", lambda x: x.quantile(0.95)),
        Mean_Total_Hours=("total_hours", "mean"),
        P_Shift_Conflict=("shift_conflicts", lambda x: (x > 0).mean()),
        Mean_Conflicts=("shift_conflicts", "mean"),
        Mean_Wet_Lease_Cost=("wet_lease_cost", "mean"),
    )
    print(summary.to_string())
    print()
    print(f"Saved {len(combined_results):,} per-simulation rows to:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()