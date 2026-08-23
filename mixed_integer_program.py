import pandas as pd

from pulp import (
    LpBinary,
    LpMinimize,
    LpStatus,
    LpVariable,
    lpSum,
    LpProblem
)


# ============================================================
# PARAMETERS
# ============================================================

MAX_STORES = 55

MAX_SKIP_PERCENT = 0.20

MAX_SKIPPED = int(
    MAX_STORES * MAX_SKIP_PERCENT
)

MAX_OWNED_SHIFTS = 40

MAX_OWNED_SHIFTS_PER_SHIFT = 20


# ============================================================
# FILE PATHS
# ============================================================

FEASIBLE_ROUTES_FILE = (
    "project_data/feasible_routes.csv"
)

OPTIMAL_ROUTES_FILE = (
    "project_data/Optimal_Routes.csv"
)

MIP_SUMMARY_FILE = (
    "project_data/MIP_Summary.csv"
)


# ============================================================
# STORE SKIP PENALTY
# ============================================================

def skip_penalty(store):

    store_clean = (
        store.lower()
        .replace("'", "")
        .replace("’", "")
        .replace(" ", "")
    )

    if "paknsave" in store_clean:
        return 1500

    return 800


# ============================================================
# MAIN MIP FUNCTION
# ============================================================

def run_mip_baseline_grouped(
    df,
    day_group_name,
    target_days,
    model_choice="C",
    silent=False
):

    df_group = df[
        df["Day"].isin(target_days)
    ].copy().reset_index(drop=True)

    # ========================================================
    # PARSE STORES
    # ========================================================

    def parse_stores(route_str):

        parts = [
            s.strip()
            for s in route_str.split("->")
        ]

        return [
            p
            for p in parts
            if p != "Warehouse"
        ]

    df_group["Visited_Stores"] = (
        df_group["Route"].apply(parse_stores)
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = LpProblem(
        f"Foodstuffs_{day_group_name}",
        LpMinimize
    )

    indices = df_group.index.tolist()

    # ========================================================
    # DECISION VARIABLES
    # ========================================================

    x = LpVariable.dicts(
        "Owned",
        indices,
        cat=LpBinary
    )

    y = LpVariable.dicts(
        "Leased",
        indices,
        cat=LpBinary
    )

    # ========================================================
    # FIND STORES
    # ========================================================

    all_stores = set()

    for stores_list in df_group["Visited_Stores"]:

        all_stores.update(stores_list)

    # ========================================================
    # SKIPPED STORE VARIABLES
    # ========================================================

    skipped = {}

    for day in target_days:

        df_day = df_group[
            df_group["Day"] == day
        ]

        day_stores = set()

        for stores_list in df_day["Visited_Stores"]:

            day_stores.update(stores_list)

        for store in day_stores:

            skipped[(day, store)] = LpVariable(
                f"Skipped_{day}_{store}",
                cat=LpBinary
            )

    # ========================================================
    # COSTS
    # ========================================================

    owned_cost = lpSum(
        df_group.loc[
            i,
            "Owned_Cost_NZD"
        ] * x[i]
        for i in indices
    )

    leased_cost = lpSum(
        df_group.loc[
            i,
            "Leased_Cost_NZD"
        ] * y[i]
        for i in indices
    )

    skipped_cost = lpSum(
        skip_penalty(store)
        * skipped[(day, store)]
        for day, store in skipped
    )

    # ========================================================
    # OBJECTIVE
    # ========================================================

    if model_choice == "M":

        STORE_PRIORITY = 1_000_000

        model += (
            STORE_PRIORITY
            * lpSum(
                skipped[(day, store)]
                for day, store in skipped
            )
            + owned_cost
            + leased_cost
        )

    else:

        model += (
            owned_cost
            + leased_cost
            + skipped_cost
        )

    # ========================================================
    # CONSTRAINTS
    # ========================================================

    for day in target_days:

        df_day = df_group[
            df_group["Day"] == day
        ]

        day_indices = df_day.index.tolist()

        # ====================================================
        # STORES
        # ====================================================

        day_stores = set()

        for stores_list in df_day["Visited_Stores"]:

            day_stores.update(stores_list)

        # ====================================================
        # EACH STORE VISITED ONCE OR SKIPPED
        # ====================================================

        for store in day_stores:

            store_routes = [
                i
                for i in day_indices
                if store in df_day.loc[
                    i,
                    "Visited_Stores"
                ]
            ]

            model += (
                lpSum(
                    x[i] + y[i]
                    for i in store_routes
                )
                + skipped[(day, store)]
                == 1
            )

        # ====================================================
        # MAXIMUM STORES SKIPPED
        # ====================================================

        if model_choice == "F":

            model += (
                lpSum(
                    skipped[(day, store)]
                    for store in day_stores
                )
                == MAX_SKIPPED
            )

        else:

            model += (
                lpSum(
                    skipped[(day, store)]
                    for store in day_stores
                )
                <= MAX_SKIPPED
            )

        # ====================================================
        # OWNED TRUCK CAPACITY
        # ====================================================

        if "Shift" in df_day.columns:

            idx_8am = df_day[
                df_day["Shift"] == "8am"
            ].index.tolist()

            idx_2pm = df_day[
                df_day["Shift"] == "2pm"
            ].index.tolist()

            model += (
                lpSum(
                    x[i]
                    for i in idx_8am
                )
                <= MAX_OWNED_SHIFTS_PER_SHIFT
            )

            model += (
                lpSum(
                    x[i]
                    for i in idx_2pm
                )
                <= MAX_OWNED_SHIFTS_PER_SHIFT
            )

        else:

            model += (
                lpSum(
                    x[i]
                    for i in day_indices
                )
                <= MAX_OWNED_SHIFTS
            )

        # ====================================================
        # ROUTE CAN ONLY BE OWNED OR LEASED
        # ====================================================

        for i in day_indices:

            model += (
                x[i] + y[i]
                <= 1
            )

    # ========================================================
    # SOLVE
    # ========================================================

    model.solve()

    status = LpStatus[model.status]

    if status != "Optimal":

        if not silent:

            print(
                f"WARNING: {day_group_name} "
                f"status = {status}"
            )

    # ========================================================
    # SELECTED ROUTES
    # ========================================================

    sel_owned = [
        i
        for i in indices
        if x[i].value() == 1
    ]

    sel_leased = [
        i
        for i in indices
        if y[i].value() == 1
    ]

    # ========================================================
    # SKIPPED STORES
    # ========================================================

    skipped_stores = {}

    for day in target_days:

        skipped_stores[day] = [

            store

            for store in all_stores

            if (
                day,
                store
            ) in skipped

            and skipped[
                (day, store)
            ].value() == 1
        ]

    # ========================================================
    # ACTUAL COST
    # ========================================================

    actual_owned_cost = sum(
        df_group.loc[
            i,
            "Owned_Cost_NZD"
        ]
        for i in sel_owned
    )

    actual_leased_cost = sum(
        df_group.loc[
            i,
            "Leased_Cost_NZD"
        ]
        for i in sel_leased
    )

    actual_skipped_cost = sum(
        skip_penalty(store)
        for day in target_days
        for store in skipped_stores[day]
    )

    actual_total_cost = (
        actual_owned_cost
        + actual_leased_cost
        + actual_skipped_cost
    )

    # ========================================================
    # VISITED STORES
    # ========================================================

    visited_stores = {}

    for day in target_days:

        visited_stores[day] = set()

        df_day = df_group[
            df_group["Day"] == day
        ]

        day_indices = df_day.index.tolist()

        for i in sel_owned + sel_leased:

            if i in day_indices:

                visited_stores[day].update(
                    df_group.loc[
                        i,
                        "Visited_Stores"
                    ]
                )

    # ========================================================
    # SELECTED ROUTES
    # ========================================================

    selected_indices = (
        sel_owned
        + sel_leased
    )

    selected_routes = df_group.loc[
        selected_indices,
        [
            "Day",
            "Route",
            "Demand",
            "Duration_hours",
            "Owned_Cost_NZD",
            "Leased_Cost_NZD"
        ]
    ].copy()

    selected_routes["Vehicle_Type"] = (
        selected_routes.index.map(
            lambda i:
            "Owned"
            if i in sel_owned
            else "Wet-Leased"
        )
    )

    selected_routes["Trip_Cost_NZD"] = (
        selected_routes.index.map(
            lambda i:

            df_group.loc[
                i,
                "Owned_Cost_NZD"
            ]

            if i in sel_owned

            else

            df_group.loc[
                i,
                "Leased_Cost_NZD"
            ]
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    if not silent:

        print()
        print(
            f"=== {day_group_name} ==="
        )

        print(
            f"Status: {status}"
        )

        print(
            f"Actual Cost: "
            f"${actual_total_cost:,.2f}"
        )

        print(
            f"Owned Shifts: "
            f"{len(sel_owned)}"
        )

        print(
            f"Wet-Leased Trips: "
            f"{len(sel_leased)}"
        )

        for day in target_days:

            print(
                f"{day}: "
                f"{len(visited_stores[day])}/55 "
                f"stores visited"
            )

            print(
                f"{day}: "
                f"{len(skipped_stores[day])}/11 "
                f"stores skipped"
            )

    return (
        actual_total_cost,
        len(sel_owned),
        len(sel_leased),
        selected_routes,
        skipped_stores
    )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    model_choice = input(
        "Do you want to run Fuel reduction "
        "(Press F), cost minimisation "
        "(Press C) or store maximisation "
        "(Press M) model): "
    ).strip().upper()

    while model_choice not in ["F", "C", "M"]:

        model_choice = input(
            "Invalid choice. Press F, C or M: "
        ).strip().upper()

    # ========================================================
    # LOAD FEASIBLE ROUTES
    # ========================================================

    print()
    print("Loading feasible routes...")

    df = pd.read_csv(
        FEASIBLE_ROUTES_FILE
    )

    print(
        f"Loaded {len(df):,} feasible routes."
    )

    # ========================================================
    # WEEKDAYS
    # ========================================================

    (
        cost_weekdays,
        owned_wd,
        leased_wd,
        routes_weekdays,
        skipped_weekdays
    ) = run_mip_baseline_grouped(
        df,
        "Weekdays",
        ["Weekdays"],
        model_choice
    )

    # ========================================================
    # SATURDAY
    # ========================================================

    (
        cost_saturdays,
        owned_sat,
        leased_sat,
        routes_saturdays,
        skipped_saturdays
    ) = run_mip_baseline_grouped(
        df,
        "Saturday",
        ["Saturday"],
        model_choice
    )

    # ============================================================
    # MISSED STORES AND PENALTIES
    # ============================================================

    # ------------------------------------------------------------
    # Number of stores missed
    # ------------------------------------------------------------

    weekday_missed_stores = len(
        skipped_weekdays["Weekdays"]
    )

    saturday_missed_stores = len(
        skipped_saturdays["Saturday"]
    )

    # ------------------------------------------------------------
    # Penalty cost for missed stores
    # ------------------------------------------------------------

    weekday_penalty_cost = sum(
        skip_penalty(store)
        for store in skipped_weekdays["Weekdays"]
    )

    saturday_penalty_cost = sum(
        skip_penalty(store)
        for store in skipped_saturdays["Saturday"]
    )

    # ------------------------------------------------------------
    # Weekly missed stores
    #
    # Weekday schedule occurs 5 times
    # ------------------------------------------------------------

    total_weekly_missed_stores = (
            weekday_missed_stores * 5
            + saturday_missed_stores
    )

    # ------------------------------------------------------------
    # Weekly penalty cost
    # ------------------------------------------------------------

    total_weekly_penalty_cost = (
            weekday_penalty_cost * 5
            + saturday_penalty_cost
    )

    # ============================================================
    # MISSED STORES AND PENALTIES
    #
    # These are reported separately for transparency.
    # They are ALREADY INCLUDED in cost_weekdays and cost_saturdays
    # through actual_total_cost in the MIP.
    # ============================================================

    # ------------------------------------------------------------
    # Number of stores missed
    # ------------------------------------------------------------

    weekday_missed_stores = len(
        skipped_weekdays["Weekdays"]
    )

    saturday_missed_stores = len(
        skipped_saturdays["Saturday"]
    )

    # ------------------------------------------------------------
    # Total weekly missed stores
    #
    # Weekdays occur 5 times per week.
    # ------------------------------------------------------------

    total_weekly_missed_stores = (
            weekday_missed_stores * 5
            + saturday_missed_stores
    )

    # ------------------------------------------------------------
    # Penalty cost for reporting
    #
    # These values are NOT added to total_weekly_cost again.
    # ------------------------------------------------------------

    weekday_penalty_cost = sum(
        skip_penalty(store)
        for store in skipped_weekdays["Weekdays"]
    )

    saturday_penalty_cost = sum(
        skip_penalty(store)
        for store in skipped_saturdays["Saturday"]
    )

    total_weekly_penalty_cost = (
            weekday_penalty_cost * 5
            + saturday_penalty_cost
    )

    # ============================================================
    # TOTAL WEEKLY COST
    #
    # cost_weekdays and cost_saturdays already include:
    #
    #   Owned route costs
    #   Wet-leased route costs
    #   Missed-store penalties
    #
    # Therefore, DO NOT add the penalty cost again.
    # ============================================================

    total_weekly_cost = (
            cost_weekdays * 5
            + cost_saturdays
    )

    # ============================================================
    # COMBINE OPTIMAL ROUTES
    # ============================================================

    optimal_routes = pd.concat(
        [
            routes_weekdays,
            routes_saturdays
        ],
        ignore_index=True
    )

    # ============================================================
    # SAVE OPTIMAL ROUTES
    # ============================================================

    optimal_routes.to_csv(
        OPTIMAL_ROUTES_FILE,
        index=False
    )

    # ============================================================
    # CREATE SUMMARY
    # ============================================================

    summary = pd.DataFrame({

        "Metric": [

            # ----------------------------------------------------
            # MODEL
            # ----------------------------------------------------

            "Model",

            # ----------------------------------------------------
            # COST
            # ----------------------------------------------------

            "Weekday Cost",
            "Saturday Cost",
            "Weekly Cost",

            # ----------------------------------------------------
            # MISSED STORES
            # ----------------------------------------------------

            "Weekday Stores Missed",
            "Saturday Stores Missed",
            "Total Weekly Stores Missed",

            # ----------------------------------------------------
            # PENALTIES
            # ----------------------------------------------------

            "Weekday Penalty Cost",
            "Saturday Penalty Cost",
            "Total Weekly Penalty Cost",

            # ----------------------------------------------------
            # VEHICLES
            # ----------------------------------------------------

            "Weekday Owned Shifts",
            "Weekday Wet-Leased Trips",

            "Saturday Owned Shifts",
            "Saturday Wet-Leased Trips",

            # ----------------------------------------------------
            # ROUTES
            # ----------------------------------------------------

            "Total Optimal Routes"
        ],

        "Value": [

            # ----------------------------------------------------
            # MODEL
            # ----------------------------------------------------

            model_choice,

            # ----------------------------------------------------
            # COST
            # ----------------------------------------------------

            cost_weekdays,
            cost_saturdays,
            total_weekly_cost,

            # ----------------------------------------------------
            # MISSED STORES
            # ----------------------------------------------------

            weekday_missed_stores,
            saturday_missed_stores,
            total_weekly_missed_stores,

            # ----------------------------------------------------
            # PENALTIES
            # ----------------------------------------------------

            weekday_penalty_cost,
            saturday_penalty_cost,
            total_weekly_penalty_cost,

            # ----------------------------------------------------
            # VEHICLES
            # ----------------------------------------------------

            owned_wd,
            leased_wd,

            owned_sat,
            leased_sat,

            # ----------------------------------------------------
            # ROUTES
            # ----------------------------------------------------

            len(optimal_routes)
        ]
    })

    # ============================================================
    # SAVE SUMMARY
    # ============================================================

    summary.to_csv(
        MIP_SUMMARY_FILE,
        index=False
    )

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print()
    print(
        "============================================================"
    )

    print(
        f"Weekday Cost: "
        f"${cost_weekdays:,.2f}"
    )

    print(
        f"Saturday Cost: "
        f"${cost_saturdays:,.2f}"
    )

    print(
        f"TOTAL WEEKLY COST: "
        f"${total_weekly_cost:,.2f}"
    )

    print()

    print(
        f"Weekday Stores Missed: "
        f"{weekday_missed_stores}"
    )

    print(
        f"Saturday Stores Missed: "
        f"{saturday_missed_stores}"
    )

    print(
        f"Total Weekly Stores Missed: "
        f"{total_weekly_missed_stores}"
    )

    print()

    print(
        f"Weekday Penalty Cost: "
        f"${weekday_penalty_cost:,.2f}"
    )

    print(
        f"Saturday Penalty Cost: "
        f"${saturday_penalty_cost:,.2f}"
    )

    print(
        f"Total Weekly Penalty Cost: "
        f"${total_weekly_penalty_cost:,.2f}"
    )

    print()

    print(
        f"Weekday Owned Shifts: "
        f"{owned_wd}"
    )

    print(
        f"Weekday Wet-Leased Trips: "
        f"{leased_wd}"
    )

    print(
        f"Saturday Owned Shifts: "
        f"{owned_sat}"
    )

    print(
        f"Saturday Wet-Leased Trips: "
        f"{leased_sat}"
    )

    print()

    print(
        f"Total Optimal Routes: "
        f"{len(optimal_routes)}"
    )

    print(
        "============================================================"
    )

    print()

    print(
        "Optimal routes saved to:"
    )

    print(
        OPTIMAL_ROUTES_FILE
    )

    print()

    print(
        "MIP summary saved to:"
    )

    print(
        MIP_SUMMARY_FILE
    )

    print(
        "============================================================"
    )