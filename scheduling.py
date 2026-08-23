import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "project_data/Optimal_Routes.csv"
OUTPUT_FILE = "project_data/Scheduled_Optimal_Routes.csv"

NUM_TRUCKS = 20

SHIFT_1_START = 8 * 60       # 8:00 AM in minutes from midnight
SHIFT_2_START = 14 * 60      # 2:00 PM in minutes from midnight

SHIFT_2_END = 24 * 60         # Midnight


# ============================================================
# LOAD LP RESULTS
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Day",
    "Route",
    "Duration_hours",
    "Vehicle_Type"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# PREPARE DATA
# ============================================================

df["Duration_hours"] = pd.to_numeric(
    df["Duration_hours"],
    errors="coerce"
)

if df["Duration_hours"].isna().any():

    raise ValueError(
        "Some routes have invalid Duration_hours values."
    )


# Convert duration to minutes

df["Duration_minutes"] = (
    df["Duration_hours"] * 60
)


# Preserve original LP order

df["Original_Index"] = df.index


# ============================================================
# TIME CONVERSION FUNCTIONS
# ============================================================

def minutes_to_time(minutes):

    hours = int(minutes // 60)
    mins = int(round(minutes % 60))

    # Handle 24:00
    if hours >= 24:
        hours = hours - 24

    return f"{hours:02d}:{mins:02d}"


def format_datetime(day, minutes):

    base_date = datetime(
        2026,
        1,
        1
    )

    return (
        base_date
        + timedelta(minutes=minutes)
    ).strftime("%H:%M")


# ============================================================
# SCHEDULING FUNCTION
# ============================================================

def schedule_day(df_day):

    # --------------------------------------------------------
    # Create truck information
    # --------------------------------------------------------

    trucks = []

    for truck_number in range(
        1,
        NUM_TRUCKS + 1
    ):

        trucks.append({

            "Truck": f"Truck {truck_number}",

            # Time when truck becomes available
            "Available_At_8am": SHIFT_1_START,

            "Available_At_2pm": SHIFT_2_START,

            # Whether truck has already been assigned
            # an 8am route
            "Has_8am_Route": False,

            # Whether truck has already been assigned
            # a 2pm route
            "Has_2pm_Route": False

        })


    # --------------------------------------------------------
    # Separate routes by starting shift
    # --------------------------------------------------------

    routes = df_day.copy()


    # --------------------------------------------------------
    # Sort longest routes first
    #
    # This makes the scheduling more robust because the
    # longest routes are the most difficult to fit.
    # --------------------------------------------------------

    routes = routes.sort_values(
        by="Duration_minutes",
        ascending=False
    ).reset_index(drop=False)


    scheduled = []


    # ========================================================
    # FIRST PASS
    #
    # Assign routes that can start at 8am.
    #
    # We try to keep the 2pm shift available whenever possible.
    # ========================================================

    for _, route in routes.iterrows():

        duration = route[
            "Duration_minutes"
        ]

        # ----------------------------------------------------
        # Determine which shift this route belongs to
        # ----------------------------------------------------

        # If the route is explicitly associated with a shift,
        # use that information.

        if "Shift" in route.index:

            shift_value = str(
                route["Shift"]
            ).lower()

        else:

            shift_value = ""


        # ----------------------------------------------------
        # Routes labelled 2pm
        # ----------------------------------------------------

        if shift_value == "2pm":

            continue


        # ----------------------------------------------------
        # Try to assign an 8am route
        # ----------------------------------------------------

        assigned = False

        for truck in trucks:

            if truck["Has_8am_Route"]:
                continue


            finish_time = (
                SHIFT_1_START
                + duration
            )


            # This truck can only receive a second shift
            # if its first route finishes by 2pm.

            # Therefore, for the 8am route, we prefer
            # routes finishing by 2pm.

            if finish_time <= SHIFT_2_START:

                truck["Has_8am_Route"] = True

                truck["Available_At_8am"] = finish_time

                scheduled.append({

                    "Original_Index":
                        route["index"],

                    "Truck":
                        truck["Truck"],

                    "Scheduled_Shift":
                        "8am",

                    "Start_Time":
                        SHIFT_1_START,

                    "End_Time":
                        finish_time,

                    "Overtime_Minutes":
                        0

                })

                assigned = True

                break


        # ----------------------------------------------------
        # If route cannot finish by 2pm, assign it to a truck
        # anyway if a truck is available.
        #
        # This means that truck loses its second shift.
        # ----------------------------------------------------

        if not assigned:

            for truck in trucks:

                if truck["Has_8am_Route"]:
                    continue


                finish_time = (
                    SHIFT_1_START
                    + duration
                )

                overtime = max(
                    0,
                    finish_time
                    - SHIFT_2_START
                )


                truck["Has_8am_Route"] = True

                truck["Available_At_8am"] = finish_time

                scheduled.append({

                    "Original_Index":
                        route["index"],

                    "Truck":
                        truck["Truck"],

                    "Scheduled_Shift":
                        "8am",

                    "Start_Time":
                        SHIFT_1_START,

                    "End_Time":
                        finish_time,

                    "Overtime_Minutes":
                        overtime

                })

                assigned = True

                break


        # ----------------------------------------------------
        # If no truck is available, leave route unscheduled
        # for the second pass.
        # ----------------------------------------------------

    # ========================================================
    # SECOND PASS
    #
    # Schedule remaining routes at 2pm.
    #
    # IMPORTANT:
    #
    # A truck can only receive a 2pm route if its 8am route
    # finished by 2pm.
    # ========================================================

    scheduled_indices = {

        item["Original_Index"]

        for item in scheduled

    }


    remaining_routes = routes[
        ~routes["index"].isin(
            scheduled_indices
        )
    ].copy()


    # --------------------------------------------------------
    # Sort remaining routes longest first
    # --------------------------------------------------------

    remaining_routes = remaining_routes.sort_values(
        by="Duration_minutes",
        ascending=False
    )


    for _, route in remaining_routes.iterrows():

        duration = route[
            "Duration_minutes"
        ]

        assigned = False


        # ----------------------------------------------------
        # Find a truck whose first shift finished by 2pm
        # ----------------------------------------------------

        for truck in trucks:

            if not truck["Has_8am_Route"]:

                # Truck has not been used in the morning.
                # It can start directly at 2pm.

                available = True

            else:

                available = (

                    truck["Available_At_8am"]
                    <= SHIFT_2_START

                )


            if not available:
                continue


            if truck["Has_2pm_Route"]:
                continue


            # ------------------------------------------------
            # Start exactly at 2pm
            # ------------------------------------------------

            start_time = SHIFT_2_START

            finish_time = (
                start_time
                + duration
            )


            overtime = max(
                0,
                finish_time
                - SHIFT_2_END
            )


            truck["Has_2pm_Route"] = True

            scheduled.append({

                "Original_Index":
                    route["index"],

                "Truck":
                    truck["Truck"],

                "Scheduled_Shift":
                    "2pm",

                "Start_Time":
                    start_time,

                "End_Time":
                    finish_time,

                "Overtime_Minutes":
                    overtime

            })

            assigned = True

            break


        # ----------------------------------------------------
        # If no truck can perform the route, it remains
        # unscheduled.
        # ----------------------------------------------------


    # ========================================================
    # RETURN SCHEDULE
    # ========================================================

    schedule_df = pd.DataFrame(
        scheduled
    )

    return (
        schedule_df,
        remaining_routes
    )


# ============================================================
# RUN SCHEDULER BY DAY
# ============================================================

all_schedules = []

unscheduled_routes = []


for day in df["Day"].unique():

    print()
    print(
        "============================================================"
    )

    print(
        f"SCHEDULING {day}"
    )

    print(
        "============================================================"
    )


    df_day = df[
        df["Day"] == day
    ].copy()


    schedule_day_df, _ = schedule_day(
        df_day
    )


    # --------------------------------------------------------
    # Identify routes that were not scheduled
    # --------------------------------------------------------

    scheduled_original_indices = set(
        schedule_day_df[
            "Original_Index"
        ]
    )


    unscheduled_day = df_day[
        ~df_day.index.isin(
            scheduled_original_indices
        )
    ].copy()


    if len(unscheduled_day) > 0:

        unscheduled_routes.append(
            unscheduled_day
        )


    # --------------------------------------------------------
    # Add day
    # --------------------------------------------------------

    schedule_day_df["Day"] = day


    all_schedules.append(
        schedule_day_df
    )


# ============================================================
# COMBINE SCHEDULE
# ============================================================

if all_schedules:

    schedule = pd.concat(
        all_schedules,
        ignore_index=True
    )

else:

    schedule = pd.DataFrame()


# ============================================================
# MERGE SCHEDULE WITH ORIGINAL ROUTES
# ============================================================

if not schedule.empty:

    schedule = schedule.merge(

        df,

        left_on="Original_Index",

        right_on="Original_Index",

        how="left",

        suffixes=(
            "",
            "_Original"
        )

    )


# ============================================================
# CONVERT TIMES TO READABLE FORMAT
# ============================================================

if not schedule.empty:

    schedule["Start_Time"] = (
        schedule["Start_Time"]
        .apply(minutes_to_time)
    )

    schedule["End_Time"] = (
        schedule["End_Time"]
        .apply(minutes_to_time)
    )


# ============================================================
# SORT SCHEDULE
# ============================================================

if not schedule.empty:

    schedule = schedule.sort_values(
        by=[
            "Day",
            "Scheduled_Shift",
            "Truck"
        ]
    )


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [

    "Day",
    "Truck",
    "Scheduled_Shift",

    "Start_Time",
    "End_Time",

    "Overtime_Minutes",

    "Route",
    "Demand",
    "Duration_hours",

    "Vehicle_Type",
    "Trip_Cost_NZD"

]


output_columns = [

    col
    for col in output_columns
    if col in schedule.columns

]


schedule = schedule[
    output_columns
]


# ============================================================
# EXPORT SCHEDULE
# ============================================================

schedule.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# EXPORT UNSCHEDULED ROUTES
# ============================================================

if unscheduled_routes:

    unscheduled = pd.concat(
        unscheduled_routes,
        ignore_index=True
    )

else:

    unscheduled = pd.DataFrame(
        columns=df.columns
    )


print("Unscheduled Routes: ", len(unscheduled_routes))

# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "============================================================"
)

print(
    "SCHEDULING SUMMARY"
)

print(
    "============================================================"
)


for day in df["Day"].unique():

    day_schedule = schedule[
        schedule["Day"] == day
    ]


    morning = day_schedule[
        day_schedule["Scheduled_Shift"]
        == "8am"
    ]

    afternoon = day_schedule[
        day_schedule["Scheduled_Shift"]
        == "2pm"
    ]


    print()
    print(day)

    print(
        f"  8am routes: "
        f"{len(morning)}"
    )

    print(
        f"  2pm routes: "
        f"{len(afternoon)}"
    )

    print(
        f"  Total routes: "
        f"{len(day_schedule)}"
    )


    # --------------------------------------------------------
    # Trucks used
    # --------------------------------------------------------

    trucks_used = (
        day_schedule["Truck"]
        .nunique()
    )

    print(
        f"  Trucks used: "
        f"{trucks_used}/{NUM_TRUCKS}"
    )


    # --------------------------------------------------------
    # Trucks doing two shifts
    # --------------------------------------------------------

    two_shift_trucks = (

        day_schedule
        .groupby("Truck")
        .size()
        .loc[lambda x: x == 2]
        .count()

    )

    print(
        f"  Trucks doing two shifts: "
        f"{two_shift_trucks}"
    )


    # --------------------------------------------------------
    # Overtime
    # --------------------------------------------------------

    overtime_routes = day_schedule[
        day_schedule[
            "Overtime_Minutes"
        ] > 0
    ]


    print(
        f"  Routes with overtime: "
        f"{len(overtime_routes)}"
    )


    if len(overtime_routes) > 0:

        print(
            "  Trucks losing second shift "
            "because of overtime:"
        )

        for truck in sorted(
            overtime_routes["Truck"].unique()
        ):

            print(
                f"    {truck}"
            )


# ============================================================
# UNSCHEDULED ROUTES
# ============================================================

print()

print(
    "============================================================"
)

print(
    f"Unscheduled Routes: "
    f"{len(unscheduled)}"
)

print(
    "============================================================"
)


if len(unscheduled) > 0:

    print(
        "\nWARNING:"
    )

    print(
        "The LP selected more routes than can be "
        "physically scheduled using 20 trucks."
    )

    print(
        "These routes are listed in "
        "'Unscheduled_Routes.csv'."
    )

else:

    print(
        "All LP-selected routes were successfully scheduled."
    )


print()

print(
    f"Schedule exported to "
    f"{OUTPUT_FILE}"
)

print(
    "Unscheduled routes exported to "
    "'Unscheduled_Routes.csv'"
)