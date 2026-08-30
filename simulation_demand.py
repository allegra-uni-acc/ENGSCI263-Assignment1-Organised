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

# --- Demand variation ------------------------------------------------------
# Day-to-day variation in pallets ordered per store, as a fraction of the
# planned (baseline) demand on that route. ADJUST to match how noisy your
# actual demand data is (e.g. std dev of Avg_Demand.csv per store/day).
DEMAND_SIGMA_FRACTION = 0.15

# --- Project constants -------------------------------------------------
UNLOAD_MINUTES_PER_PALLET = 18.0
TRUCK_COST_PER_HOUR = 220.0
OVERTIME_THRESHOLD_HOURS = 4.0
OVERTIME_COST_PER_HOUR = 310.0
WET_LEASE_COST_PER_2H_BLOCK = 1400.0
TRUCK_CAPACITY = 16.0  # pallets -- a truck physically cannot carry more

TRUCKS_PER_SHIFT = 20
SHIFT_TARGET_HOURS = 3.5

# SIMPLIFICATION: a truck whose AM route finishes only slightly after 14:00
# is treated as NOT needing a wet-lease substitute for its PM route -- e.g.
# the driver making up a few minutes, or the 2pm departure having a little
# real-world flex. Only overruns PAST this grace period count as a genuine
# conflict. This softens the (very real) razor-thin baseline slack on
# several trucks in the schedule -- it does not change that underlying
# fragility, just how much of it this simulation treats as costly.
# ADJUST or set to 0 to go back to a strict "any overrun = conflict" model.
CONFLICT_GRACE_PERIOD_MINUTES = 15.0


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

        # A truck can only physically carry TRUCK_CAPACITY pallets. Demand
        # above that on a route can't ride along with the scheduled truck --
        # it needs a SEPARATE wet-lease truck sent for the overflow pallets.
        # ASSUMPTION: since this simulation works at route level (not
        # per-store), the overflow truck's travel time is approximated as
        # the same as the original route's baseline driving time (it's
        # heading to the same area) -- only its unload time is specific to
        # the overflow amount. Adjust this if you have per-store demand
        # detail to route the overflow truck more precisely.
        capped_demand = np.minimum(demand_sim, TRUCK_CAPACITY)
        overflow_demand = np.maximum(0.0, demand_sim - TRUCK_CAPACITY)
        unload_hours_sim = capped_demand * UNLOAD_MINUTES_PER_PALLET / 60.0

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

        # ---- overflow wet-lease trucks for demand exceeding capacity ----
        overflow_cost = 0.0
        n_overflow_events = int((overflow_demand > 0).sum())
        if n_overflow_events > 0:
            overflow_hours = baseline_drive_hours + overflow_demand * UNLOAD_MINUTES_PER_PALLET / 60.0
            overflow_hours = overflow_hours[overflow_demand > 0]
            overflow_blocks = np.ceil(overflow_hours / 2.0 - 1e-12)
            overflow_cost = float((overflow_blocks * WET_LEASE_COST_PER_2H_BLOCK).sum())

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

        total_cost = costs.sum() + overflow_cost

        rows.append({
            "simulation": sim,
            "total_hours": actual_hours.sum(),
            "routes_over_3_5h": int((actual_hours > 3.5).sum()),
            "routes_over_4h": int((actual_hours > 4.0).sum()),
            "shift_conflicts": n_conflicts,
            "wet_lease_cost": wet_lease_cost,
            "capacity_overflow_events": n_overflow_events,
            "capacity_overflow_cost": overflow_cost,
            "total_cost": total_cost,
        })

    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, day_label: str, n_routes: int) -> pd.DataFrame:
    return pd.DataFrame({
        "metric": [
            "Routes scheduled", "Mean total cost", "Median total cost",
            "95th pct total cost", "Mean total truck-hours",
            "Mean # routes > 3.5h", "P(any route > 4h)",
            "P(a shift-conflict occurs)", "Mean # conflicts per sim",
            "Mean shift-conflict wet lease cost",
            "P(a capacity-overflow occurs)", "Mean # overflow events per sim",
            "Mean overflow wet lease cost",
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
            (results["capacity_overflow_events"] > 0).mean(),
            results["capacity_overflow_events"].mean(),
            results["capacity_overflow_cost"].mean(),
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