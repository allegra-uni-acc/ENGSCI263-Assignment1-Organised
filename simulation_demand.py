# ============================================================
# BOOTSTRAPPING DEMAND + ROUTING + MIP  (PARALLELIZED)
# ============================================================
#
# Same methodology as the original script -- only the execution
# strategy changed: bootstrap iterations are independent of each
# other, so they're run across multiple processes instead of one
# at a time. See the bottom of this file for other speed-ups that
# require changes to route_generation.py / mixed_integer_program.py.
#
# ============================================================

import importlib
import os
import time
import concurrent.futures

import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

DEMAND_FILE = "project_data/FoodstuffsDemand2026.csv"
DURATIONS_FILE = "project_data/FoodstuffsDurations2026.csv"
LOCATIONS_FILE = "project_data/FoodstuffsLocations.csv"

ROUTING_MODULE_NAME = "route_generation"
MIP_MODULE_NAME = "mixed_integer_program"

N_BOOTSTRAPS = 1000
RANDOM_SEED = 263
Z_VALUE = 1.96

# Leave None to auto-detect (uses all logical cores). Set an explicit
# number if you want to leave some cores free for other work.
N_WORKERS = None

OUTPUT_DIR = "project_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD + PREPARE DEMAND  (unchanged from your original)
# ============================================================

def load_raw_demand():
    data = pd.read_csv(DEMAND_FILE)
    print(f"Loaded {data['Supermarket'].nunique()} stores and "
          f"{len(data.columns) - 1} daily observations.")
    return data


def prepare_demand_data(data):
    data_melt = data.melt(id_vars="Supermarket", var_name="Date", value_name="Demand")
    data_melt["Date"] = pd.to_datetime(data_melt["Date"], dayfirst=True)
    data_melt["Day"] = data_melt["Date"].dt.day_name()

    data_melt["Chain"] = ""
    data_melt.loc[data_melt["Supermarket"].str.startswith("Four"), "Chain"] = "Four Square"
    data_melt.loc[data_melt["Supermarket"].str.startswith("New"), "Chain"] = "New World"
    data_melt.loc[data_melt["Supermarket"].str.startswith("Pak"), "Chain"] = "PAK'nSAVE"

    data_melt = data_melt[data_melt["Day"] != "Sunday"]
    data_melt = data_melt[
        (data_melt["Date"] != pd.Timestamp("2026-06-01"))
        & (data_melt["Demand"] < 20)
    ]

    data_melt["Period"] = ""
    data_melt.loc[
        data_melt["Day"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        "Period",
    ] = "Weekdays"
    data_melt.loc[data_melt["Day"] == "Saturday", "Period"] = "Saturday"
    data_melt = data_melt[data_melt["Period"] != ""].copy()

    return data_melt


def calculate_demand_simplified(data_melt):
    mean = data_melt.groupby(["Supermarket", "Period"])["Demand"].mean()
    std_dev = data_melt.groupby(["Supermarket", "Period"])["Demand"].std()
    upper = mean + Z_VALUE * std_dev
    estimate = np.ceil(upper).astype(int)

    warehouse_rows = pd.Series(
        0,
        index=pd.MultiIndex.from_product(
            [["Warehouse"], estimate.index.get_level_values("Period").unique()],
            names=["Supermarket", "Period"],
        ),
    )
    estimate = pd.concat([estimate, warehouse_rows])
    return estimate.rename("Demand").reset_index()


# ============================================================
# BOOTSTRAP DEMAND  (unchanged)
# ============================================================

def bootstrap_demand(data_melt, rng):
    bootstrap_rows = []
    grouped = data_melt.groupby(["Supermarket", "Period"])

    for (supermarket, period), group in grouped:
        values = group["Demand"].dropna().to_numpy()
        bootstrap_sample = rng.choice(values, size=len(values), replace=True)
        mean = np.mean(bootstrap_sample)
        std_dev = np.std(bootstrap_sample, ddof=1) if len(bootstrap_sample) > 1 else 0.0
        upper = mean + Z_VALUE * std_dev
        demand_estimate = int(np.ceil(upper))
        bootstrap_rows.append({"Supermarket": supermarket, "Period": period, "Demand": demand_estimate})

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    warehouse_rows = pd.DataFrame({
        "Supermarket": ["Warehouse", "Warehouse"],
        "Period": ["Weekdays", "Saturday"],
        "Demand": [0, 0],
    })
    bootstrap_df = pd.concat([bootstrap_df, warehouse_rows], ignore_index=True)
    bootstrap_df["Demand"] = bootstrap_df["Demand"].astype(int)
    return bootstrap_df


def validate_bootstrap_demand(bootstrap_df):
    required_columns = {"Supermarket", "Period", "Demand"}
    missing_columns = required_columns - set(bootstrap_df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in bootstrap demand: {missing_columns}")

    periods = set(bootstrap_df["Period"])
    if periods != {"Weekdays", "Saturday"}:
        raise ValueError(f"Unexpected periods: {periods}")

    non_warehouse = bootstrap_df[bootstrap_df["Supermarket"] != "Warehouse"]
    n_stores = non_warehouse["Supermarket"].nunique()
    if n_stores != 55:
        raise ValueError(f"Expected 55 stores, found {n_stores}.")

    warehouse = bootstrap_df[bootstrap_df["Supermarket"] == "Warehouse"]
    if len(warehouse) != 2:
        raise ValueError("Expected two Warehouse rows.")
    if not (warehouse["Demand"] == 0).all():
        raise ValueError("Warehouse demand must be zero.")


# ============================================================
# MODULE LOADING -- done ONCE PER WORKER PROCESS, not per bootstrap
# ============================================================

# Populated once per worker process by _init_worker; each worker
# process gets its own copy of these globals.
_routing_module = None
_mip_module = None


def _load_modules():
    routing_module = importlib.import_module(ROUTING_MODULE_NAME)
    mip_module = importlib.import_module(MIP_MODULE_NAME)

    if not hasattr(routing_module, "generate_feasible_routes"):
        raise AttributeError(
            "\nYour routing file does not contain:\n\ngenerate_feasible_routes()\n\n"
            f"Expected file:\n{ROUTING_MODULE_NAME}.py"
        )
    if not hasattr(mip_module, "run_mip_baseline_grouped"):
        raise AttributeError(
            "\nYour MIP file does not contain:\n\nrun_mip_baseline_grouped()\n\n"
            f"Expected file:\n{MIP_MODULE_NAME}.py"
        )
    return routing_module, mip_module


def _init_worker():
    """Runs once when each worker process starts -- imports the
    routing/MIP modules a single time per process instead of once
    per bootstrap iteration."""
    global _routing_module, _mip_module
    _routing_module, _mip_module = _load_modules()


# ============================================================
# ROUTING + MIP  (unchanged logic)
# ============================================================

def run_routing(routing_module, bootstrap_df):
    return routing_module.generate_feasible_routes(
        demand_source=bootstrap_df,
        durations_file=DURATIONS_FILE,
        locations_file=LOCATIONS_FILE,
        silent=True,
    )


def run_mip(mip_module, routes_df):
    (weekday_cost, weekday_owned, weekday_leased,
     weekday_routes, weekday_skipped) = mip_module.run_mip_baseline_grouped(
        routes_df, "Weekdays", ["Weekdays"], model_choice="C", silent=True,
    )
    (saturday_cost, saturday_owned, saturday_leased,
     saturday_routes, saturday_skipped) = mip_module.run_mip_baseline_grouped(
        routes_df, "Saturday", ["Saturday"], model_choice="C", silent=True,
    )

    weekly_cost = weekday_cost * 5 + saturday_cost

    weekday_skipped_count = sum(len(s) for s in weekday_skipped.values())
    saturday_skipped_count = sum(len(s) for s in saturday_skipped.values())

    weekday_route_demand = weekday_routes["Demand"].sum() if not weekday_routes.empty else 0
    saturday_route_demand = saturday_routes["Demand"].sum() if not saturday_routes.empty else 0
    total_route_demand = weekday_route_demand + saturday_route_demand

    return {
        "Weekday_Cost": weekday_cost,
        "Saturday_Cost": saturday_cost,
        "Weekly_Cost": weekly_cost,
        "Total_Route_Demand": total_route_demand,
        "Weekday_Owned": weekday_owned,
        "Weekday_Leased": weekday_leased,
        "Saturday_Owned": saturday_owned,
        "Saturday_Leased": saturday_leased,
        "Weekday_Skipped": weekday_skipped_count,
        "Saturday_Skipped": saturday_skipped_count,
    }


# ============================================================
# ONE BOOTSTRAP -- runs inside a worker process
# ============================================================

def _run_one_bootstrap(args):
    bootstrap_number, data_melt, seed = args
    rng = np.random.default_rng(seed)

    bootstrap_df = bootstrap_demand(data_melt, rng)
    validate_bootstrap_demand(bootstrap_df)

    routes_df = run_routing(_routing_module, bootstrap_df)
    mip_result = run_mip(_mip_module, routes_df)

    return {
        "Bootstrap": bootstrap_number,
        "Weekday_Cost": mip_result["Weekday_Cost"],
        "Saturday_Cost": mip_result["Saturday_Cost"],
        "Weekly_Cost": mip_result["Weekly_Cost"],
        "Total_Route_Demand": mip_result["Total_Route_Demand"],
        "Weekday_Owned": mip_result["Weekday_Owned"],
        "Weekday_Leased": mip_result["Weekday_Leased"],
        "Saturday_Owned": mip_result["Saturday_Owned"],
        "Saturday_Leased": mip_result["Saturday_Leased"],
        "Weekday_Skipped": mip_result["Weekday_Skipped"],
        "Saturday_Skipped": mip_result["Saturday_Skipped"],
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    start_time = time.time()

    print("Loading raw demand data...")
    raw_data = load_raw_demand()
    data_melt = prepare_demand_data(raw_data)
    print(f"After cleaning: {len(data_melt)} observations.")

    simplified_demand = calculate_demand_simplified(data_melt)

    n_workers = N_WORKERS or os.cpu_count()

    print()
    print("=" * 60)
    print("BOOTSTRAPPING DEMAND (parallel)")
    print("=" * 60)
    print(f"Bootstrap samples: {N_BOOTSTRAPS}")
    print(f"Worker processes: {n_workers}")
    print("Method: mean + 1.96 x sample SD, rounded up")
    print("Bootstrap level: Store x Period")
    print("Periods: Weekdays / Saturday")
    print("=" * 60)

    # One independent, high-quality seed per bootstrap so workers
    # don't produce correlated random draws.
    seed_sequence = np.random.SeedSequence(RANDOM_SEED)
    child_seeds = seed_sequence.spawn(N_BOOTSTRAPS)

    tasks = [
        (b, data_melt, child_seeds[b - 1])
        for b in range(1, N_BOOTSTRAPS + 1)
    ]

    results = []
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker
    ) as executor:
        futures = {executor.submit(_run_one_bootstrap, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            bootstrap_number = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Bootstrap {bootstrap_number} failed: {exc}")
                raise
            results.append(result)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == N_BOOTSTRAPS:
                elapsed = time.time() - start_time
                print(f"Completed {completed}/{N_BOOTSTRAPS}... ({elapsed:.1f}s elapsed)")

    # restore bootstrap order (as_completed finishes out of order)
    results.sort(key=lambda r: r["Bootstrap"])
    results_df = pd.DataFrame(results)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "simulation_demand_results.csv"), index=False)

    # ---- cost summary ----
    costs = results_df["Weekly_Cost"]
    summary = pd.DataFrame({
        "Statistic": ["Mean", "Median", "Standard Deviation", "5th Percentile",
                      "95th Percentile", "Minimum", "Maximum"],
        "Weekly_Cost_NZD": [
            costs.mean(), costs.median(), costs.std(),
            np.percentile(costs, 5), np.percentile(costs, 95),
            costs.min(), costs.max(),
        ],
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "simulation_demand_summary.csv"), index=False)

    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("BOOTSTRAP COMPLETE")
    print("=" * 60)
    print(f"Successful bootstraps: {len(results_df)}")
    print(f"Total time: {total_time:.1f}s ({total_time / N_BOOTSTRAPS:.2f}s per bootstrap avg)")
    print()
    print(f"Mean weekly cost: ${costs.mean():,.2f}")
    print(f"Median weekly cost: ${costs.median():,.2f}")
    print(f"Standard deviation: ${costs.std():,.2f}")
    print()
    print(f"5th percentile: ${np.percentile(costs, 5):,.2f}")
    print(f"95th percentile: ${np.percentile(costs, 95):,.2f}")
    print()
    print("Files saved:")
    print("  simulation_demand_results.csv")
    print("  simulation_demand_summary.csv")
    print("=" * 60)