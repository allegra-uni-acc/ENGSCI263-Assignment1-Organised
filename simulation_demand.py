# ============================================================
# BOOTSTRAPPING DEMAND + ROUTING + MIP
# ============================================================
#
# ROUTING FILE:
#   route_generation.py
#
# MIP FILE:
#   mixed_integer_program.py
#
# DEMAND METHOD:
#
#   1. Remove Sundays
#   2. Remove 1 June 2026
#   3. Remove demand >= 20
#   4. Group Monday-Friday as "Weekdays"
#   5. Group Saturday as "Saturday"
#   6. Bootstrap observations within each store × period
#   7. Calculate:
#
#          mean + 1.96 × sample SD
#
#   8. Round UP using np.ceil()
#
#   9. Pass this demand directly into routing
#  10. Solve the MIP
#
# ============================================================


import importlib
import os
import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

DEMAND_FILE = "project_data/FoodstuffsDemand2026.csv"

DURATIONS_FILE = "project_data/FoodstuffsDurations2026.csv"

LOCATIONS_FILE = "project_data/FoodstuffsLocations.csv"


# ============================================================
# EXACT MODULE NAMES
# ============================================================

ROUTING_MODULE_NAME = (
    "route_generation"
)

MIP_MODULE_NAME = (
    "mixed_integer_program"
)


# ============================================================
# BOOTSTRAP PARAMETERS
# ============================================================

N_BOOTSTRAPS = 1000

RANDOM_SEED = 263

Z_VALUE = 1.96

# ============================================================
# OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = "project_data"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD RAW DEMAND
# ============================================================

def load_raw_demand():

    data = pd.read_csv(
        DEMAND_FILE
    )

    print(
        f"Loaded "
        f"{data['Supermarket'].nunique()} stores "
        f"and "
        f"{len(data.columns) - 1} daily observations."
    )

    return data


# ============================================================
# PREPARE DEMAND
#
# THIS MATCHES THE PREPROCESSING USED TO CREATE
# demand_estimate.csv
# ============================================================

def prepare_demand_data(
    data
):

    # --------------------------------------------------------
    # Melt data
    # --------------------------------------------------------

    data_melt = data.melt(
        id_vars="Supermarket",
        var_name="Date",
        value_name="Demand"
    )

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    data_melt["Date"] = pd.to_datetime(
        data_melt["Date"],
        dayfirst=True
    )

    # --------------------------------------------------------
    # Day of week
    # --------------------------------------------------------

    data_melt["Day"] = (
        data_melt["Date"]
        .dt
        .day_name()
    )

    # --------------------------------------------------------
    # Chain
    #
    # Not required for the bootstrap, but retained to match
    # your original preprocessing.
    # --------------------------------------------------------

    data_melt["Chain"] = ""

    data_melt.loc[
        data_melt["Supermarket"]
        .str.startswith("Four"),
        "Chain"
    ] = "Four Square"

    data_melt.loc[
        data_melt["Supermarket"]
        .str.startswith("New"),
        "Chain"
    ] = "New World"

    data_melt.loc[
        data_melt["Supermarket"]
        .str.startswith("Pak"),
        "Chain"
    ] = "PAK'nSAVE"

    # --------------------------------------------------------
    # REMOVE SUNDAYS
    #
    # Same as:
    #
    # data_melt = data_melt[
    #     data_melt["Day"] != "Sunday"
    # ]
    # --------------------------------------------------------

    data_melt = data_melt[
        data_melt["Day"] != "Sunday"
    ]

    # --------------------------------------------------------
    # REMOVE 1 JUNE 2026
    #
    # Same as:
    #
    # data_melt = data_melt[
    #     (data_melt["Date"] != "2026-06-01")
    #     & (data_melt["Demand"] < 20)
    # ]
    # --------------------------------------------------------

    data_melt = data_melt[
        (
            data_melt["Date"]
            != pd.Timestamp("2026-06-01")
        )
        &
        (
            data_melt["Demand"] < 20
        )
    ]

    # --------------------------------------------------------
    # CREATE PERIOD
    #
    # Monday-Friday -> Weekdays
    # Saturday      -> Saturday
    # --------------------------------------------------------

    data_melt["Period"] = ""

    data_melt.loc[
        data_melt["Day"].isin(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday"
            ]
        ),
        "Period"
    ] = "Weekdays"

    data_melt.loc[
        data_melt["Day"] == "Saturday",
        "Period"
    ] = "Saturday"

    # --------------------------------------------------------
    # Remove anything that wasn't assigned to a period
    # --------------------------------------------------------

    data_melt = data_melt[
        data_melt["Period"] != ""
    ].copy()

    return data_melt


# ============================================================
# CREATE NORMAL Demand_Simplified DATASET
#
# This is included so we can verify that the calculation
# matches your original demand_estimate.csv method.
# ============================================================

def calculate_demand_simplified(
    data_melt
):

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    mean = (
        data_melt
        .groupby(
            [
                "Supermarket",
                "Period"
            ]
        )["Demand"]
        .mean()
    )

    # --------------------------------------------------------
    # Sample standard deviation
    #
    # pandas .std() uses ddof=1.
    # --------------------------------------------------------

    std_dev = (
        data_melt
        .groupby(
            [
                "Supermarket",
                "Period"
            ]
        )["Demand"]
        .std()
    )

    # --------------------------------------------------------
    # Upper 95% bound
    # --------------------------------------------------------

    upper = (
        mean
        + Z_VALUE * std_dev
    )

    # --------------------------------------------------------
    # Round UP
    # --------------------------------------------------------

    estimate = (
        np.ceil(upper)
        .astype(int)
    )

    # --------------------------------------------------------
    # Add Warehouse
    # --------------------------------------------------------

    warehouse_rows = pd.Series(
        0,
        index=pd.MultiIndex.from_product(
            [
                ["Warehouse"],
                estimate
                .index
                .get_level_values(
                    "Period"
                )
                .unique()
            ],
            names=[
                "Supermarket",
                "Period"
            ]
        )
    )

    estimate = pd.concat(
        [
            estimate,
            warehouse_rows
        ]
    )

    estimate = (
        estimate
        .rename("Demand")
        .reset_index()
    )

    return estimate


# ============================================================
# BOOTSTRAP DEMAND
#
# IMPORTANT:
#
# We bootstrap the ORIGINAL cleaned observations for each
# store × period.
#
# For each bootstrap sample:
#
#     mean = sample mean
#     SD   = sample SD (ddof=1)
#
#     upper = mean + 1.96 × SD
#
#     demand = ceil(upper)
#
# This is the SAME calculation used in demand_estimate.csv.
# ============================================================

def bootstrap_demand(
    data_melt,
    rng
):

    bootstrap_rows = []

    # --------------------------------------------------------
    # Group by store and period
    #
    # This means:
    #
    # Four Square Botany Junction + Weekdays
    # Four Square Botany Junction + Saturday
    # etc.
    #
    # Weekdays are pooled together exactly as in
    # demand_estimate.csv.
    # --------------------------------------------------------

    grouped = data_melt.groupby(
        [
            "Supermarket",
            "Period"
        ]
    )

    for (
        supermarket,
        period
    ), group in grouped:

        values = (
            group["Demand"]
            .dropna()
            .to_numpy()
        )

        # ----------------------------------------------------
        # Bootstrap sample
        #
        # Same number of observations as the original group.
        # Sampling is WITH replacement.
        # ----------------------------------------------------

        bootstrap_sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        # ----------------------------------------------------
        # SAMPLE MEAN
        # ----------------------------------------------------

        mean = np.mean(
            bootstrap_sample
        )

        # ----------------------------------------------------
        # SAMPLE STANDARD DEVIATION
        #
        # ddof=1 is critical because pandas:
        #
        # .groupby()["Demand"].std()
        #
        # uses sample SD.
        # ----------------------------------------------------

        if len(bootstrap_sample) > 1:

            std_dev = np.std(
                bootstrap_sample,
                ddof=1
            )

        else:

            std_dev = 0.0

        # ----------------------------------------------------
        # EXACT SAME FORMULA
        # ----------------------------------------------------

        upper = (
            mean
            + Z_VALUE * std_dev
        )

        # ----------------------------------------------------
        # EXACT SAME ROUNDING
        # ----------------------------------------------------

        demand_estimate = int(
            np.ceil(upper)
        )

        bootstrap_rows.append(
            {
                "Supermarket":
                    supermarket,

                "Period":
                    period,

                "Demand":
                    demand_estimate
            }
        )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    # --------------------------------------------------------
    # Add Warehouse rows
    # --------------------------------------------------------

    warehouse_rows = pd.DataFrame(
        {
            "Supermarket":
                [
                    "Warehouse",
                    "Warehouse"
                ],

            "Period":
                [
                    "Weekdays",
                    "Saturday"
                ],

            "Demand":
                [
                    0,
                    0
                ]
        }
    )

    bootstrap_df = pd.concat(
        [
            bootstrap_df,
            warehouse_rows
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Ensure integer demand
    # --------------------------------------------------------

    bootstrap_df["Demand"] = (
        bootstrap_df["Demand"]
        .astype(int)
    )

    return bootstrap_df


# ============================================================
# VALIDATE BOOTSTRAP DEMAND
# ============================================================

def validate_bootstrap_demand(
    bootstrap_df
):

    required_columns = {
        "Supermarket",
        "Period",
        "Demand"
    }

    missing_columns = (
        required_columns
        -
        set(bootstrap_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing columns in bootstrap "
            f"demand: {missing_columns}"
        )

    # --------------------------------------------------------
    # Check periods
    # --------------------------------------------------------

    periods = set(
        bootstrap_df["Period"]
    )

    if periods != {
        "Weekdays",
        "Saturday"
    }:

        raise ValueError(
            "Unexpected periods: "
            f"{periods}"
        )

    # --------------------------------------------------------
    # Check number of stores
    # --------------------------------------------------------

    non_warehouse = (
        bootstrap_df[
            bootstrap_df["Supermarket"]
            != "Warehouse"
        ]
    )

    n_stores = (
        non_warehouse[
            "Supermarket"
        ]
        .nunique()
    )

    if n_stores != 55:

        raise ValueError(
            "Expected 55 stores, "
            f"found {n_stores}."
        )

    # --------------------------------------------------------
    # Check Warehouse
    # --------------------------------------------------------

    warehouse = bootstrap_df[
        bootstrap_df["Supermarket"]
        == "Warehouse"
    ]

    if len(warehouse) != 2:

        raise ValueError(
            "Expected two Warehouse rows."
        )

    if not (
        warehouse["Demand"] == 0
    ).all():

        raise ValueError(
            "Warehouse demand must be zero."
        )


# ============================================================
# LOAD ROUTING MODULE
# ============================================================

def load_modules():

    print()
    print(
        "Loading routing and MIP..."
    )

    # --------------------------------------------------------
    # Exact routing filename
    # Python module names do not include .py
    # --------------------------------------------------------

    routing_module = importlib.import_module(
        "Truck_Routing_18_Minute_Issue_Solved_lol"
    )

    print(
        "Routing loaded."
    )

    # --------------------------------------------------------
    # Exact MIP filename
    # --------------------------------------------------------

    mip_module = importlib.import_module(
        "mixed_integer_program"
    )

    print(
        "MIP loaded."
    )

    # --------------------------------------------------------
    # Check expected functions
    # --------------------------------------------------------

    if not hasattr(
        routing_module,
        "generate_feasible_routes"
    ):

        raise AttributeError(
            "\nYour routing file does not "
            "contain:\n\n"
            "generate_feasible_routes()\n\n"
            "Expected file:\n"
            "route_generation.py"
        )

    if not hasattr(
        mip_module,
        "run_mip_baseline_grouped"
    ):

        raise AttributeError(
            "\nYour MIP file does not "
            "contain:\n\n"
            "run_mip_baseline_grouped()\n\n"
            "Expected file:\n"
            "mixed_integer_program.py"
        )

    return (
        routing_module,
        mip_module
    )


# ============================================================
# RUN ROUTING
# ============================================================

def run_routing(
    routing_module,
    bootstrap_df
):

    routes_df = (
        routing_module
        .generate_feasible_routes(
            demand_source=bootstrap_df,

            durations_file=
                DURATIONS_FILE,

            locations_file=
                LOCATIONS_FILE,

            silent=True
        )
    )

    return routes_df


# ============================================================
# RUN MIP
# ============================================================

def run_mip(
    mip_module,
    routes_df
):

    # --------------------------------------------------------
    # WEEKDAYS
    # --------------------------------------------------------

    (
        weekday_cost,
        weekday_owned,
        weekday_leased,
        weekday_routes,
        weekday_skipped
    ) = (
        mip_module
        .run_mip_baseline_grouped(
            routes_df,

            "Weekdays",

            ["Weekdays"],

            model_choice="C",

            silent=True
        )
    )

    # --------------------------------------------------------
    # SATURDAY
    # --------------------------------------------------------

    (
        saturday_cost,
        saturday_owned,
        saturday_leased,
        saturday_routes,
        saturday_skipped
    ) = (
        mip_module
        .run_mip_baseline_grouped(
            routes_df,

            "Saturday",

            ["Saturday"],

            model_choice="C",

            silent=True
        )
    )

    # --------------------------------------------------------
    # WEEKLY COST
    #
    # Weekdays occur 5 times.
    # Saturday occurs once.
    # --------------------------------------------------------

    weekly_cost = (
        weekday_cost * 5
        + saturday_cost
    )

    # --------------------------------------------------------
    # Count skipped stores
    # --------------------------------------------------------

    weekday_skipped_count = sum(
        len(
            stores
        )
        for stores
        in weekday_skipped.values()
    )

    saturday_skipped_count = sum(
        len(
            stores
        )
        for stores
        in saturday_skipped.values()
    )

    # --------------------------------------------------------
    # TOTAL DEMAND IN SELECTED OPTIMAL ROUTES
    #
    # This is the sum of the demand carried by all routes
    # selected by the MIP for Weekdays + Saturday.
    #
    # This is NOT multiplied by 5 because we are comparing
    # the demand represented by the optimal route solution
    # directly to your chosen total demand of 674.
    # --------------------------------------------------------

    weekday_route_demand = (
        weekday_routes["Demand"].sum()
        if not weekday_routes.empty
        else 0
    )

    saturday_route_demand = (
        saturday_routes["Demand"].sum()
        if not saturday_routes.empty
        else 0
    )

    total_route_demand = (
            weekday_route_demand
            + saturday_route_demand
    )

    return {

        "Weekday_Cost":
            weekday_cost,

        "Saturday_Cost":
            saturday_cost,

        "Weekly_Cost":
            weekly_cost,

        "Total_Route_Demand":
            total_route_demand,

        "Weekday_Owned":
            weekday_owned,

        "Weekday_Leased":
            weekday_leased,

        "Saturday_Owned":
            saturday_owned,

        "Saturday_Leased":
            saturday_leased,

        "Weekday_Skipped":
            weekday_skipped_count,

        "Saturday_Skipped":
            saturday_skipped_count
    }


# ============================================================
# RUN ONE BOOTSTRAP
# ============================================================

def run_one_bootstrap(
    bootstrap_number,
    data_melt,
    routing_module,
    mip_module,
    rng
):

    # --------------------------------------------------------
    # Generate bootstrap demand
    # --------------------------------------------------------

    bootstrap_df = bootstrap_demand(
        data_melt,
        rng
    )

    validate_bootstrap_demand(
        bootstrap_df
    )

    # --------------------------------------------------------
    # Generate feasible routes
    # --------------------------------------------------------

    routes_df = run_routing(
        routing_module,
        bootstrap_df
    )

    # --------------------------------------------------------
    # Solve MIP
    # --------------------------------------------------------

    mip_result = run_mip(
        mip_module,
        routes_df
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    result = {

        "Bootstrap":
            bootstrap_number,

        "Weekday_Cost":
            mip_result[
                "Weekday_Cost"
            ],

        "Saturday_Cost":
            mip_result[
                "Saturday_Cost"
            ],

        "Weekly_Cost":
            mip_result[
                "Weekly_Cost"
            ],

        "Total_Route_Demand":
            mip_result[
                "Total_Route_Demand"
            ],

        "Weekday_Owned":
            mip_result[
                "Weekday_Owned"
            ],

        "Weekday_Leased":
            mip_result[
                "Weekday_Leased"
            ],

        "Saturday_Owned":
            mip_result[
                "Saturday_Owned"
            ],

        "Saturday_Leased":
            mip_result[
                "Saturday_Leased"
            ],

        "Weekday_Skipped":
            mip_result[
                "Weekday_Skipped"
            ],

        "Saturday_Skipped":
            mip_result[
                "Saturday_Skipped"
            ]
    }

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # LOAD RAW DATA
    # --------------------------------------------------------

    print(
        "Loading raw demand data..."
    )

    raw_data = load_raw_demand()

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    data_melt = prepare_demand_data(
        raw_data
    )

    print(
        f"After cleaning: "
        f"{len(data_melt)} observations."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Check that the NON-BOOTSTRAPPED calculation exactly
    # matches your demand_estimate.csv methodology.
    #
    # This does not get used for the bootstrap itself.
    # It is simply a verification/reference.
    # --------------------------------------------------------

    simplified_demand = (
        calculate_demand_simplified(
            data_melt
        )
    )

    # --------------------------------------------------------
    # LOAD MODULES
    # --------------------------------------------------------

    (
        routing_module,
        mip_module
    ) = load_modules()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )
    print(
        "BOOTSTRAPPING DEMAND"
    )
    print(
        "=" * 60
    )

    print(
        f"Bootstrap samples: "
        f"{N_BOOTSTRAPS}"
    )

    print(
        "Method: mean + 1.96 × sample SD, "
        "rounded up"
    )

    print(
        "Bootstrap level: "
        "Store × Period"
    )

    print(
        "Periods: "
        "Weekdays / Saturday"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # RANDOM NUMBER GENERATOR
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # RUN BOOTSTRAPS
    # --------------------------------------------------------

    for b in range(
        1,
        N_BOOTSTRAPS + 1
    ):

        if (
            b == 1
            or b % 10 == 0
            or b == N_BOOTSTRAPS
        ):

            print(
                f"Running bootstrap "
                f"{b}/{N_BOOTSTRAPS}..."
            )

        result = run_one_bootstrap(
            b,
            data_melt,
            routing_module,
            mip_module,
            rng
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # SAVE ALL BOOTSTRAP RESULTS
    # --------------------------------------------------------

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "simulation_demand_results.csv"
        ),
        index=False
    )

    # ========================================================
    # COST SUMMARY
    # ========================================================

    costs = (
        results_df[
            "Weekly_Cost"
        ]
    )

    mean_cost = costs.mean()

    median_cost = costs.median()

    std_cost = costs.std()

    percentile_5 = np.percentile(
        costs,
        5
    )

    percentile_95 = np.percentile(
        costs,
        95
    )

    minimum_cost = costs.min()

    maximum_cost = costs.max()

    # --------------------------------------------------------
    # Summary dataframe
    # --------------------------------------------------------

    summary = pd.DataFrame(
        {
            "Statistic":
                [
                    "Mean",
                    "Median",
                    "Standard Deviation",
                    "5th Percentile",
                    "95th Percentile",
                    "Minimum",
                    "Maximum"
                ],

            "Weekly_Cost_NZD":
                [
                    mean_cost,
                    median_cost,
                    std_cost,
                    percentile_5,
                    percentile_95,
                    minimum_cost,
                    maximum_cost
                ]
        }
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "simulation_demand_summary.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "BOOTSTRAP COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Successful bootstraps: "
        f"{len(results_df)}"
    )

    print()

    print(
        f"Mean weekly cost: "
        f"${mean_cost:,.2f}"
    )

    print(
        f"Median weekly cost: "
        f"${median_cost:,.2f}"
    )

    print(
        f"Standard deviation: "
        f"${std_cost:,.2f}"
    )

    print()

    print(
        f"5th percentile: "
        f"${percentile_5:,.2f}"
    )

    print(
        f"95th percentile: "
        f"${percentile_95:,.2f}"
    )

    print()

    print(
        f"Minimum: "
        f"${minimum_cost:,.2f}"
    )

    print(
        f"Maximum: "
        f"${maximum_cost:,.2f}"
    )

    print()

    print(
        "Files saved:"
    )

    print(
        "  simulation_demand_results.csv"
    )

    print(
        "  simulation_demand_summary.csv"
    )

    print(
        "=" * 60
    )