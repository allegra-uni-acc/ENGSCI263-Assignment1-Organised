import math
import os
import pandas as pd
from itertools import permutations


# ============================================================
# FILE PATHS
# ============================================================

PROJECT_DATA_DIR = "project_data"

DEFAULT_DEMAND_FILE = os.path.join(
    PROJECT_DATA_DIR,
    "demand_estimate.csv"
)

DEFAULT_DURATIONS_FILE = os.path.join(
    PROJECT_DATA_DIR,
    "FoodstuffsDurations2026.csv"
)

DEFAULT_LOCATIONS_FILE = os.path.join(
    PROJECT_DATA_DIR,
    "FoodstuffsLocations.csv"
)

DEFAULT_OUTPUT_FILE = os.path.join(
    PROJECT_DATA_DIR,
    "feasible_routes.csv"
)


# ============================================================
# COSTING
# ============================================================

STANDARD_SHIFT_SECONDS = 4 * 60 * 60

OVERTIME_RATE_PER_HOUR = 310.00

OWNED_RATE_PER_HOUR = 220.00

LEASED_BLOCK_HOURS = 2

LEASED_BLOCK_RATE = 1400.00


def owned_cost(
    duration_seconds,
    day=None
):

    normal_seconds = min(
        duration_seconds,
        STANDARD_SHIFT_SECONDS
    )

    overtime_seconds = max(
        0,
        duration_seconds - STANDARD_SHIFT_SECONDS
    )

    normal_hours = (
        normal_seconds / 3600
    )

    overtime_hours_billed = (
        math.ceil(
            overtime_seconds / 3600
        )
        if overtime_seconds > 0
        else 0
    )

    normal_cost = (
        normal_hours
        * OWNED_RATE_PER_HOUR
    )

    overtime_cost = (
        overtime_hours_billed
        * OVERTIME_RATE_PER_HOUR
    )

    return round(
        normal_cost + overtime_cost,
        2
    )


def leased_cost(
    duration_seconds
):

    hours = (
        duration_seconds / 3600
    )

    blocks = math.ceil(
        hours / LEASED_BLOCK_HOURS
    )

    return round(
        blocks * LEASED_BLOCK_RATE,
        2
    )


# ============================================================
# CLASSES
# ============================================================

class Store:

    def __init__(
        self,
        name,
        demand,
        location
    ):

        self.name = name
        self.demand = demand
        self.location = location


class Route:

    def __init__(
        self,
        stores,
        demand,
        duration
    ):

        self.stores = stores
        self.demand = demand
        self.duration = duration


class Network:

    def __init__(
        self,
        duration_matrix
    ):

        self.duration_matrix = (
            duration_matrix
        )

        self.stores = {}

    def add_store(
        self,
        name,
        demand,
        location
    ):

        self.stores[name] = Store(
            name,
            demand,
            location
        )


# ============================================================
# ROUTE GENERATION PARAMETERS
# ============================================================

MAX_DURATION = (
    7 * 60 * 60
)

UNLOAD_TIME_PER_PALLET = (
    18 * 60
)

K_NEAREST = 6

TRUCK_CAPACITY = 16


# ============================================================
# MAIN ROUTE-GENERATION FUNCTION
# ============================================================

def generate_feasible_routes(
    demand_source=DEFAULT_DEMAND_FILE,
    durations_file=DEFAULT_DURATIONS_FILE,
    locations_file=DEFAULT_LOCATIONS_FILE,
    silent=False
):

    # ========================================================
    # LOAD DATA
    # ========================================================

    if not silent:

        print()
        print(
            "Loading routing data..."
        )

    # --------------------------------------------------------
    # Durations
    # --------------------------------------------------------

    Durations = pd.read_csv(
        durations_file
    )

    # --------------------------------------------------------
    # Locations
    # --------------------------------------------------------

    Locations = pd.read_csv(
        locations_file
    )

    # --------------------------------------------------------
    # Demand
    #
    # The bootstrap passes a DataFrame directly.
    #
    # Otherwise load the supplied CSV.
    # --------------------------------------------------------

    if isinstance(
        demand_source,
        pd.DataFrame
    ):

        Demand = (
            demand_source
            .copy()
        )

    else:

        Demand = pd.read_csv(
            demand_source
        )

    # ========================================================
    # PREPARE DURATION MATRIX
    # ========================================================

    if "Unnamed: 0" in Durations.columns:

        Durations = Durations.rename(
            columns={
                "Unnamed: 0":
                    "Origin"
            }
        )

    elif "Origin" not in Durations.columns:

        raise ValueError(
            "Could not find the Origin column "
            "in the duration file."
        )

    Durations = Durations.set_index(
        "Origin"
    )

    # --------------------------------------------------------
    # Ensure duration matrix is numeric
    # --------------------------------------------------------

    Durations = Durations.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ========================================================
    # PREPARE DEMAND DATA
    # ========================================================

    # Expected format:
    #
    # Supermarket | Period | Demand
    #
    # Period:
    #   Weekdays
    #   Saturday

    if "Period" not in Demand.columns:

        # ----------------------------------------------------
        # Allow raw wide demand data
        # ----------------------------------------------------

        if "Supermarket" not in Demand.columns:

            raise ValueError(
                "Demand data must contain "
                "'Supermarket'."
            )

        date_columns = [
            c
            for c in Demand.columns
            if c != "Supermarket"
        ]

        long_rows = []

        for _, row in Demand.iterrows():

            for date in date_columns:

                date_value = pd.to_datetime(
                    date,
                    dayfirst=True
                )

                weekday = (
                    date_value.weekday()
                )

                if weekday == 5:

                    period = "Saturday"

                else:

                    period = "Weekdays"

                long_rows.append(
                    {
                        "Supermarket":
                            row[
                                "Supermarket"
                            ],

                        "Period":
                            period,

                        "Demand":
                            row[date]
                    }
                )

        raw_long = pd.DataFrame(
            long_rows
        )

        Demand = (
            raw_long
            .groupby(
                [
                    "Supermarket",
                    "Period"
                ],
                as_index=False
            )["Demand"]
            .mean()
        )

    # ========================================================
    # VALIDATE DEMAND FORMAT
    # ========================================================

    required_demand_columns = {
        "Supermarket",
        "Period",
        "Demand"
    }

    missing = (
        required_demand_columns
        - set(Demand.columns)
    )

    if missing:

        raise ValueError(
            "Demand data is missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Make sure demand is numeric
    # --------------------------------------------------------

    Demand["Demand"] = pd.to_numeric(
        Demand["Demand"],
        errors="coerce"
    )

    # ========================================================
    # CREATE DEMAND MATRIX
    # ========================================================

    demand_wide = Demand.pivot(
        index="Supermarket",
        columns="Period",
        values="Demand"
    )

    # --------------------------------------------------------
    # Ensure both periods exist
    # --------------------------------------------------------

    if "Weekdays" not in demand_wide.columns:

        demand_wide["Weekdays"] = 0

    if "Saturday" not in demand_wide.columns:

        demand_wide["Saturday"] = 0

    demand_wide = demand_wide[
        [
            "Weekdays",
            "Saturday"
        ]
    ]

    demand_wide.columns = [
        "Demand_Weekdays",
        "Demand_Saturday"
    ]

    demand_wide = (
        demand_wide
        .reset_index()
    )

    # ========================================================
    # MERGE LOCATIONS AND DEMAND
    # ========================================================

    if "Supermarket" not in Locations.columns:

        raise ValueError(
            "Locations file must contain "
            "'Supermarket'."
        )

    supermarket_info = (
        Locations.merge(
            demand_wide,
            on="Supermarket",
            how="left"
        )
    )

    # --------------------------------------------------------
    # Check missing demand
    # --------------------------------------------------------

    missing_demand = (
        supermarket_info[
            supermarket_info[
                [
                    "Demand_Weekdays",
                    "Demand_Saturday"
                ]
            ]
            .isna()
            .any(axis=1)
        ]
    )

    if len(missing_demand) > 0:

        if not silent:

            print(
                "WARNING - no demand data "
                "found for:"
            )

            print(
                list(
                    missing_demand[
                        "Supermarket"
                    ]
                )
            )

        supermarket_info[
            [
                "Demand_Weekdays",
                "Demand_Saturday"
            ]
        ] = (
            supermarket_info[
                [
                    "Demand_Weekdays",
                    "Demand_Saturday"
                ]
            ]
            .fillna(0)
        )

    # ========================================================
    # BUILD NETWORK
    # ========================================================

    network = Network(
        Durations
    )

    # --------------------------------------------------------
    # Warehouse
    # --------------------------------------------------------

    network.add_store(
        "Warehouse",

        {
            "Weekdays": 0,
            "Saturday": 0
        },

        (0, 0)
    )

    # --------------------------------------------------------
    # Stores
    # --------------------------------------------------------

    for i in range(
        len(supermarket_info)
    ):

        name = (
            supermarket_info
            .loc[
                i,
                "Supermarket"
            ]
        )

        demand = {

            "Weekdays":
                float(
                    supermarket_info.loc[
                        i,
                        "Demand_Weekdays"
                    ]
                ),

            "Saturday":
                float(
                    supermarket_info.loc[
                        i,
                        "Demand_Saturday"
                    ]
                )
        }

        location = (
            supermarket_info.loc[
                i,
                "Lat"
            ],
            supermarket_info.loc[
                i,
                "Long"
            ]
        )

        network.add_store(
            name,
            demand,
            location
        )

    # ========================================================
    # CHECK DURATION MATRIX
    # ========================================================

    route_order = list(
        network.duration_matrix.index
    )

    required_stores = list(
        network.stores.keys()
    )

    missing_from_duration = [
        store
        for store in required_stores
        if store not in route_order
    ]

    if missing_from_duration:

        raise ValueError(
            "The following stores are missing "
            "from the duration matrix:\n"
            f"{missing_from_duration}"
        )

    # --------------------------------------------------------
    # Ensure all duration matrix destinations exist
    # --------------------------------------------------------

    missing_columns = [
        store
        for store in required_stores
        if store not in Durations.columns
    ]

    if missing_columns:

        raise ValueError(
            "The following stores are missing "
            "as columns from the duration matrix:\n"
            f"{missing_columns}"
        )

    # ========================================================
    # FAST DURATION LOOKUP
    # ========================================================

    route_position = {
        name: i
        for i, name
        in enumerate(route_order)
    }

    duration_array = (
        network
        .duration_matrix
        .loc[
            route_order,
            route_order
        ]
        .values
    )

    def travel_time(
        a,
        b
    ):

        return duration_array[
            route_position[a],
            route_position[b]
        ]

    # ========================================================
    # K-NEAREST NEIGHBOURS
    # ========================================================

    knn = {}

    for store in route_order:

        if store == "Warehouse":
            continue

        others = [
            s
            for s in route_order
            if s not in (
                "Warehouse",
                store
            )
        ]

        others.sort(
            key=lambda s:
                travel_time(
                    store,
                    s
                )
        )

        knn[store] = set(
            others[:K_NEAREST]
        )

    def is_candidate_neighbor(
        a,
        b
    ):

        return (
            b in knn.get(a, set())
            or
            a in knn.get(b, set())
        )

    # ========================================================
    # OPTIMAL STOP ORDERING
    # ========================================================

    travel_cache = {}

    def best_travel_order(
        stops,
        warehouse
    ):

        key = frozenset(
            stops
        )

        if key in travel_cache:

            return travel_cache[key]

        stops = list(
            stops
        )

        # ----------------------------------------------------
        # Single-store route
        # ----------------------------------------------------

        if len(stops) == 1:

            store = stops[0]

            total = (
                travel_time(
                    warehouse,
                    store
                )
                +
                travel_time(
                    store,
                    warehouse
                )
            )

            result = (
                [store],
                total
            )

            travel_cache[key] = result

            return result

        # ----------------------------------------------------
        # Find best permutation
        # ----------------------------------------------------

        best_order = None
        best_time = None

        for perm in permutations(
            stops
        ):

            total = travel_time(
                warehouse,
                perm[0]
            )

            for a, b in zip(
                perm,
                perm[1:]
            ):

                total += travel_time(
                    a,
                    b
                )

            total += travel_time(
                perm[-1],
                warehouse
            )

            if (
                best_time is None
                or total < best_time
            ):

                best_time = total
                best_order = perm

        result = (
            list(best_order),
            best_time
        )

        travel_cache[key] = result

        return result

    # ========================================================
    # ROUTE DURATION
    # ========================================================

    def optimal_route_duration(
        stops,
        warehouse,
        day
    ):

        order, travel = (
            best_travel_order(
                stops,
                warehouse
            )
        )

        unload_time = sum(

            network
            .stores[store]
            .demand[day]
            * UNLOAD_TIME_PER_PALLET

            for store in stops

        )

        total_duration = (
            travel
            + unload_time
        )

        return (
            order,
            total_duration
        )

    # ========================================================
    # GENERATE ROUTES
    # ========================================================

    def generate_routes(
        warehouse,
        capacity,
        day
    ):

        routes = []

        stores = [
            store
            for store
            in network.stores.keys()
            if store != warehouse
        ]

        store_position = {
            name: i
            for i, name
            in enumerate(stores)
        }

        def search(
            route,
            visited,
            demand,
            order,
            duration
        ):

            # ------------------------------------------------
            # Save current route
            # ------------------------------------------------

            if visited:

                routes.append(
                    Route(
                        [warehouse]
                        + order
                        + [warehouse],

                        demand,

                        duration
                    )
                )

                if (
                    not silent
                    and
                    len(routes) % 2000 == 0
                ):

                    print(
                        f"  ...{len(routes)} "
                        "routes found so far"
                    )

            # ------------------------------------------------
            # Try adding another store
            # ------------------------------------------------

            for store in stores:

                if store in visited:
                    continue

                # ------------------------------------------------
                # Avoid reverse duplicate route generation
                # ------------------------------------------------

                if route:

                    previous = route[-1]

                    if (
                        store_position[store]
                        <
                        store_position[previous]
                    ):

                        continue

                    if not is_candidate_neighbor(
                        previous,
                        store
                    ):

                        continue

                # ------------------------------------------------
                # New demand
                # ------------------------------------------------

                store_demand = (
                    network
                    .stores[store]
                    .demand[day]
                )

                new_demand = (
                    demand
                    + store_demand
                )

                if new_demand > capacity:
                    continue

                # ------------------------------------------------
                # New set of stores
                # ------------------------------------------------

                new_visited = (
                    visited
                    | {store}
                )

                # ------------------------------------------------
                # Calculate optimal route duration
                # ------------------------------------------------

                (
                    new_order,
                    new_duration
                ) = optimal_route_duration(
                    new_visited,
                    warehouse,
                    day
                )

                # ------------------------------------------------
                # Duration constraint
                # ------------------------------------------------

                if (
                    new_duration
                    > MAX_DURATION
                ):

                    continue

                # ------------------------------------------------
                # Continue search
                # ------------------------------------------------

                search(
                    route + [store],

                    new_visited,

                    new_demand,

                    new_order,

                    new_duration
                )

        search(
            [],
            set(),
            0,
            [],
            0
        )

        return routes

    # ========================================================
    # GENERATE ROUTES FOR EACH PERIOD
    # ========================================================

    days = [
        "Weekdays",
        "Saturday"
    ]

    all_routes = []

    route_counter = 0

    for day in days:

        if not silent:

            print()
            print(
                "Generating routes for:",
                day
            )

        routes = generate_routes(
            "Warehouse",
            capacity=TRUCK_CAPACITY,
            day=day
        )

        if not silent:

            print(
                "Routes generated:",
                len(routes)
            )

        # ----------------------------------------------------
        # Convert routes to dataframe rows
        # ----------------------------------------------------

        for i, route in enumerate(
            routes
        ):

            route_counter += 1

            all_routes.append(
                {

                    "Global_Route_ID":
                        route_counter,

                    "Day":
                        day,

                    "Route_ID":
                        i + 1,

                    "Route":
                        " -> ".join(
                            route.stores
                        ),

                    "Demand":
                        route.demand,

                    "Duration_seconds":
                        route.duration,

                    "Duration_hours":
                        route.duration
                        / 3600,

                    "Overtime":
                        route.duration
                        > STANDARD_SHIFT_SECONDS,

                    "Owned_Cost_NZD":
                        owned_cost(
                            route.duration,
                            day
                        ),

                    "Leased_Cost_NZD":
                        leased_cost(
                            route.duration
                        )
                }
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    routes_df = pd.DataFrame(
        all_routes
    )

    # --------------------------------------------------------
    # Sort for consistency
    # --------------------------------------------------------

    if not routes_df.empty:

        routes_df = (
            routes_df
            .sort_values(
                [
                    "Day",
                    "Route_ID"
                ]
            )
            .reset_index(drop=True)
        )

    return routes_df


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "============================================"
    )

    print(
        "GENERATING FEASIBLE ROUTES"
    )

    print(
        "============================================"
    )

    print(
        f"Demand file: "
        f"{DEFAULT_DEMAND_FILE}"
    )

    print(
        f"Durations file: "
        f"{DEFAULT_DURATIONS_FILE}"
    )

    print(
        f"Locations file: "
        f"{DEFAULT_LOCATIONS_FILE}"
    )

    # --------------------------------------------------------
    # Create output directory if needed
    # --------------------------------------------------------

    os.makedirs(
        PROJECT_DATA_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate routes
    # --------------------------------------------------------

    routes_df = generate_feasible_routes(
        demand_source=DEFAULT_DEMAND_FILE,

        durations_file=
            DEFAULT_DURATIONS_FILE,

        locations_file=
            DEFAULT_LOCATIONS_FILE,

        silent=False
    )

    # --------------------------------------------------------
    # Save routes
    # --------------------------------------------------------

    routes_df.to_csv(
        DEFAULT_OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "============================================"
    )

    print(
        "FINISHED GENERATING ROUTES"
    )

    print(
        "============================================"
    )

    print(
        "Total routes:",
        len(routes_df)
    )

    if not routes_df.empty:

        print()

        print(
            "Routes by period:"
        )

        print(
            routes_df[
                "Day"
            ]
            .value_counts()
        )

        print()

        print(
            "Demand range:",
            routes_df[
                "Demand"
            ].min(),
            "to",
            routes_df[
                "Demand"
            ].max()
        )

        print(
            "Duration range:",
            round(
                routes_df[
                    "Duration_hours"
                ].min(),
                2
            ),
            "to",
            round(
                routes_df[
                    "Duration_hours"
                ].max(),
                2
            ),
            "hours"
        )

    print()

    print(
        "Saved:"
    )

    print(
        DEFAULT_OUTPUT_FILE
    )

    print(
        "============================================"
    )

    print()

    print(
        routes_df.head()
    )