from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# =========================

ROUTE_FILE = Path("project_data/optimal_routes.csv")

N_SIMULATIONS = 1000
RANDOM_SEED = 263

# Model variation around baseline traffic
DAY_TO_DAY_SIGMA = 0.30
ROUTE_TO_ROUTE_SIGMA = 0.15

# Project constants
UNLOAD_MINUTES_PER_PALLET = 18.0
TRUCK_COST_PER_HOUR = 220.0
OVERTIME_THRESHOLD_HOURS = 4.0
OVERTIME_COST_PER_BLOCK = 310.0
WET_LEASE_COST_PER_2H_BLOCK = 1400.0

# 20 owned trucks × 2 shifts × 3.5 hours
NORMAL_TRUCK_HOURS = 20 * 2 * 3.5


def parse_route(route_text: str) -> List[str]:
    """Split route into individual stops."""
    return [
        part.strip()
        for part in str(route_text).split("->")
        if part.strip()
    ]


def get_pallet_count(row: pd.Series, route_stops: List[str]) -> float:
    """
    Get the pallet count for the route.

    If a pallet/demand column exists, use it.
    Otherwise, estimate one pallet per store stop.
    """

    for col in ["Pallets", "Demand", "Total_Pallets", "Volume"]:
        if col in row and not pd.isna(row[col]):
            return float(row[col])

    # Exclude Warehouse from the stop count
    return float(
        sum(
            1
            for stop in route_stops
            if stop.strip().lower() != "warehouse"
        )
    )


def ceil_hours_over_threshold(
    total_hours: float,
    threshold: float
) -> int:
    """Calculate overtime blocks beyond the threshold."""

    extra = total_hours - threshold

    if extra <= 0:
        return 0

    return int(math.ceil(extra - 1e-12))


def simulate_offline(
    routes: pd.DataFrame,
    n_simulations: int
) -> pd.DataFrame:

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    # ----------------------------------------------------------
    # Prepare ALL routes once
    # ----------------------------------------------------------

    pallets = []
    unload_hours = []
    base_drive_hours = []

    for _, row in routes.iterrows():

        stops = parse_route(row["Route"])

        route_pallets = get_pallet_count(
            row,
            stops
        )

        pallets.append(route_pallets)

        # 18 minutes per pallet
        route_unload_hours = (
            route_pallets
            * UNLOAD_MINUTES_PER_PALLET
            / 60.0
        )

        unload_hours.append(route_unload_hours)

        # Remove unloading time from the original route duration
        route_drive_hours = max(
            0.0,
            float(row["Duration_hours"])
            - route_unload_hours
        )

        base_drive_hours.append(route_drive_hours)

    pallets = np.array(pallets)
    unload_hours = np.array(unload_hours)
    base_drive_hours = np.array(base_drive_hours)

    number_of_routes = len(routes)

    print(
        f"Simulating {number_of_routes:,} feasible routes "
        f"across {n_simulations:,} simulations..."
    )

    # ----------------------------------------------------------
    # Run simulations
    # ----------------------------------------------------------

    for sim in range(1, n_simulations + 1):

        # Common traffic conditions for the day
        day_multiplier = rng.lognormal(
            mean=-0.5 * DAY_TO_DAY_SIGMA ** 2,
            sigma=DAY_TO_DAY_SIGMA
        )

        # Independent traffic variation between routes
        route_multiplier = rng.lognormal(
            mean=-0.5 * ROUTE_TO_ROUTE_SIGMA ** 2,
            sigma=ROUTE_TO_ROUTE_SIGMA,
            size=number_of_routes
        )

        # Overall traffic factor for each route
        traffic_factor = (
            day_multiplier
            * route_multiplier
        )

        # ------------------------------------------------------
        # Calculate actual duration of every route
        # ------------------------------------------------------

        actual_hours = (
            base_drive_hours
            * traffic_factor
            + unload_hours
        )

        # ------------------------------------------------------
        # Route statistics
        # ------------------------------------------------------

        total_actual_hours = actual_hours.sum()

        average_route_hours = (
            total_actual_hours
            / number_of_routes
        )

        routes_over_35 = np.sum(
            actual_hours > 3.5
        )

        routes_over_4 = np.sum(
            actual_hours > 4.0
        )

        # ------------------------------------------------------
        # Overtime
        # ------------------------------------------------------

        overtime_blocks_per_route = np.array([
            ceil_hours_over_threshold(
                duration,
                OVERTIME_THRESHOLD_HOURS
            )
            for duration in actual_hours
        ])

        overtime_blocks = (
            overtime_blocks_per_route.sum()
        )

        overtime_cost = (
            overtime_blocks
            * OVERTIME_COST_PER_BLOCK
        )

        # ------------------------------------------------------
        # Normal truck cost
        # ------------------------------------------------------

        normal_route_cost = (
            total_actual_hours
            * TRUCK_COST_PER_HOUR
        )

        # ------------------------------------------------------
        # Wet leasing
        # ------------------------------------------------------

        extra_hours = max(
            0.0,
            total_actual_hours
            - NORMAL_TRUCK_HOURS
        )

        if extra_hours > 0:
            wet_lease_blocks = int(
                math.ceil(
                    extra_hours / 2.0
                    - 1e-12
                )
            )
        else:
            wet_lease_blocks = 0

        wet_lease_cost = (
            wet_lease_blocks
            * WET_LEASE_COST_PER_2H_BLOCK
        )

        # ------------------------------------------------------
        # Total cost
        # ------------------------------------------------------

        total_cost = (
            normal_route_cost
            + overtime_cost
            + wet_lease_cost
        )

        rows.append(
            {
                "simulation": sim,
                "total_actual_hours": total_actual_hours,
                "avg_route_hours": average_route_hours,
                "routes_over_3_5h": routes_over_35,
                "routes_over_4h": routes_over_4,
                "overtime_blocks": overtime_blocks,
                "overtime_cost": overtime_cost,
                "extra_hours_beyond_140": extra_hours,
                "wet_lease_blocks": wet_lease_blocks,
                "wet_lease_cost": wet_lease_cost,
                "normal_truck_cost": normal_route_cost,
                "total_cost": total_cost,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:

    # ----------------------------------------------------------
    # Load ALL feasible routes
    # ----------------------------------------------------------

    if not ROUTE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {ROUTE_FILE.resolve()}"
        )

    routes = pd.read_csv(ROUTE_FILE)

    print(
        f"Loaded {len(routes):,} feasible routes."
    )

    # ----------------------------------------------------------
    # Run simulation
    # ----------------------------------------------------------

    results = simulate_offline(
        routes=routes,
        n_simulations=N_SIMULATIONS
    )

    # ----------------------------------------------------------
    # Display summary
    # ----------------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "Simulations",
                "Feasible routes",
                "Mean daily cost",
                "Median daily cost",
                "95th percentile daily cost",
                "Mean total route hours",
                "Mean routes over 3.5h",
                "Probability any route > 4h",
                "Probability wet lease required",
                "Mean wet lease cost",
                "Mean overtime cost",
            ],

            "value": [
                N_SIMULATIONS,
                len(routes),

                results["total_cost"].mean(),
                results["total_cost"].median(),
                results["total_cost"].quantile(0.95),

                results["total_actual_hours"].mean(),

                results["routes_over_3_5h"].mean(),

                (
                    results["routes_over_4h"] > 0
                ).mean(),

                (
                    results["wet_lease_blocks"] > 0
                ).mean(),

                results["wet_lease_cost"].mean(),

                results["overtime_cost"].mean(),
            ],
        }
    )

    print("\n==============================")
    print("SIMULATION RESULTS")
    print("==============================")

    print(
        summary.to_string(index=False)
    )


if __name__ == "__main__":
    main()