# compile everything we know into connections of working papers
#   - to Recommendations
#   - to Agenda Items
#   - to past meetings
#   - to past WPs
#   - to future meetings

import pandas as pd
import os
import csv

# paramters
# ---

this_meeting = "XII"

# where input data is saved
dname_in = "../../results/norag_2_claude_api/"

# about working papers
fname_wp2ai = "wp_to_agenda_1.csv"
fname_wp2rec = "wp_to_rec_4.csv"
fname_wpsum = "wp_summary_2.csv"

# about agenda items
fname_ai = "agenda_summary_2.csv"

# where to save
dname_out = dname_in
fname_out_wp = "working_paper_connections_2.csv"
fname_out_ai = "agenda_item_connections_2.csv"


# fix Agenda Items
# ---

# sometimes agenda items get merged in the minutes, and so we'll need to merge
# them in the mapping from wp -> Agenda Item
df_wp2ai = pd.read_csv(
    os.path.join(dname_in, fname_wp2ai), index_col="working_paper_id"
)
df_wp2ai["agenda_item_numbers"] = df_wp2ai["agenda_item_numbers"].astype(str)

df_ai = pd.read_csv(os.path.join(dname_in, fname_ai), index_col="agenda_item_nbr")
if df_ai.index.str.contains(" | ").any():
    # some agenda items have been merged
    nbrsV = [s for s in df_ai.index if " | " in s]

    # create a mapping from a single Agenda Item number to its merged form
    D = dict()
    for nbrs in nbrsV:
        nbrV = nbrs.split(" | ")
        for nbr in nbrV:
            D[nbr] = nbrs

    # read in wp2ai and remap Agenda Item numbers
    for nbr, nbrs in D.items():
        idxs = df_wp2ai[df_wp2ai["agenda_item_numbers"] == nbr].index
        df_wp2ai.loc[idxs, "agenda_item_numbers"] = nbrs


# create mapping from Agenda Items to outcomes
# ---

ai_connections = {
    ai: {
        "agenda_item_nbr": ai,
        "title": df_ai.loc[ai]["title"],
        "wp_ids": " | ".join(df_wp2ai[df_wp2ai["agenda_item_numbers"] == ai].index),
        "rec_ids": (
            " | ".join([
                s
                for s in df_ai.loc[ai]["which_recommendations"].split(" | ")
                if this_meeting in s
            ])
            if type(df_ai.loc[ai]["which_recommendations"]) is str
            else ""
        ),
        "withdrawn": df_ai.loc[ai]["withdrawn_recommendations"],
        "past_meets": df_ai.loc[ai]["past_discussions_referenced"],
        "future_meets": df_ai.loc[ai]["future_meetings_referenced"],
        "next_meet": df_ai.loc[ai]["future_meetings_includes_next"],
    }
    for ai in df_ai.index
}

# write to csv
field_names = [
    "agenda_item_nbr",
    "title",
    "wp_ids",
    "rec_ids",
    "withdrawn",
    "past_meets",
    "future_meets",
    "next_meet",
]
with open(os.path.join(dname_out, fname_out_ai), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for output_dict in ai_connections.values():
        writer.writerow(output_dict)


# create Working Paper connections
# ---

# import each working-paper data file
df_wp2rec = pd.read_csv(os.path.join(dname_in, fname_wp2rec))
df_wpsum = pd.read_csv(
    os.path.join(dname_in, fname_wpsum), index_col="working_paper_id"
)

# create a dictionary of connections WP -> Rec
wps_that_contrib = set(df_wp2rec[df_wp2rec["contributed"]]["working_paper_id"])
wp2rec = {wp: set() for wp in wps_that_contrib}

for wp, rec, contributed in zip(
    df_wp2rec["working_paper_id"],
    df_wp2rec["recommendation_id"],
    df_wp2rec["contributed"],
):
    if contributed:
        wp2rec[wp].add(rec)

wp_connections = {
    wp: {
        "working_paper_id": wp,
        "title": df_wpsum.loc[wp]["full_title"],
        "rec_ids": " | ".join(wp2rec[wp]) if wp in wp2rec else None,
        "past_meets": df_wpsum.loc[wp]["previous_meeting_referenced"],
        "future_meets": df_wpsum.loc[wp]["future_meeting_referenced"],
        "next_meet": df_wpsum.loc[wp]["future_meetings_includes_next"],
        "past_papers": df_wpsum.loc[wp]["nbr_of_previous_papers_referenced"] > 0,
    }
    for wp in df_wpsum.index
}

# write to csv
field_names = [
    "working_paper_id",
    "title",
    "rec_ids",
    "past_papers",
    "past_meets",
    "future_meets",
    "next_meet",
]
with open(os.path.join(dname_out, fname_out_wp), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for output_dict in wp_connections.values():
        writer.writerow(output_dict)
