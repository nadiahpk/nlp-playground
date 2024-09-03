# plot how the cosine similarities between WPs and Recs compare to
# the reality of which WP contributed to which Rec

import os
import pandas as pd
import matplotlib.pyplot as plt

# parameters
# ---

# where are similarities stored
dname_in = "../../results/norag_2_claude_api/"
fname_in = "similarities_voyage_law_2.csv"

# where information is about working papers and recommendations
fname_wp2rec = "wp_to_rec_4.csv" # WP to Recommendations from LLM
fname_excl = "wp_unlikely_recs.csv"

# where to save plot
dname_out = dname_in
fname_out_fnc = lambda rec: f"similarities_rec_{rec}_voyage_law_2.pdf" # noqa

# by hand, the relationships I checked for false positive / negative
# and whether I agreed or disagreed with Claude's decision
#
# format: (rec, wp, Agree/Disagree/Uncertain) 
by_hand = [
    (1, 15, "Agree"),
    (1, 23, "Agree"),
    (5, 17, "Agree"),
    (5, 14, "Agree"),
    (2, 22, "Agree"),
    (2, 5, "Uncertain"),
    (2, 28, "Agree"),
    (3, 25, "Disagree"),
    (3, 6, "Agree"),
    (3, 26, "Agree"),
    (4, 18, "Agree"),
    (4, 9, "Uncertain"),
    (4, 6, "Uncertain"),
    (4, 1, "Agree"),
    (6, 10, "Agree"),
    (6, 19, "Agree"),
    (6, 20, "Agree"),
    (6, 9, "Agree"),
    (6, 23, "Agree"),
    (6, 28, "Agree"),
    (6, 21, "Uncertain"),
    (6, 11, "Uncertain"),
    (7, 16, "Agree"),
    (7, 5, "Agree"),
    (7, 22, "Agree"),
    (8, 19, "Uncertain"),
    (8, 20, "Uncertain"),
    (8, 26, "Agree"),
]


# which WPs contributed to each Recommendation
# ---

df = pd.read_csv(os.path.join(dname_in, fname_wp2rec))
df = df[df["contributed"]]
recs = set(df["recommendation_id"])
rec2wps = {
    rec: set(df[df["recommendation_id"] == rec]["working_paper_id"])
    for rec in recs
}

# which WPs were excluded at the summary level?
# ---

df_excl = pd.read_csv(os.path.join(dname_in, fname_excl), index_col="working_paper_id")
wp2excl = {
    wp: set(df_excl.loc[wp]["did_not_contrib_to"].split(" | "))
    for wp in df_excl.index
}
rec2excl = {rec: set() for rec in recs}
for wp, excl_recs in wp2excl.items():
    for rec in excl_recs:
        rec2excl[rec].add(wp)


# import similarities data
df = pd.read_csv(os.path.join(dname_in, fname_in), index_col=0)

for rec in df.columns:
    rec_nbr = rec.split("-")[1]

    # which WPs contributed
    contributeds = [s.split("/")[2] for s in rec2wps[rec]]

    # which WPs excluded
    excludeds = [s.split("/")[2] for s in rec2excl[rec]]

    # get and sort the working papers high to low
    wps_sims_conts = sorted(list(zip(df.index, df[rec])), key = lambda v: v[1], reverse=True)
    wps, sims = zip(*wps_sims_conts)
    # wps = [s.split("/")[2] for s in wps]
    wp_shorts = [s.split("/")[2] for s in wps]

    # plot it
    plt.figure(figsize=(7, 2))

    """
    bar_colours1 = ["red" if wp in rec2wps[rec] else "blue" for wp in wps]
    bar_colours2 = ["black" if wp in rec2excl[rec] else "green" for wp in wps]
    bar_colours = ["black" if c2 == "black" else c1 for c1, c2 in zip(bar_colours1, bar_colours2)]
    plt.bar(wp_shorts, sims, color = bar_colours, alpha=0.6)
    """
    bar_colours = ["red" if wp in rec2wps[rec] else "blue" for wp in wps]
    bar_alphas = [0.5 if wp in rec2excl[rec] else 0.8 for wp in wps]
    plt.bar(wp_shorts, sims, color = list(zip(bar_colours, bar_alphas)))

    # add by_hand annotations

    # filter relevant
    relevant = [v for v in by_hand if str(v[0]) == rec_nbr]
    if relevant:
        _, wp_checks, results = zip(*relevant)
        idx_checks = [wp_shorts.index(str(wp)) for wp in wp_checks]
        for idx_check, result in zip(idx_checks, results):
            plt.text(idx_check, sims[idx_check], result[:1], ha="center", va="bottom")



    # plt.ylabel(f"cosine similarity to Rec. {rec_nbr}")
    # plt.xlabel('Working Paper')
    plt.ylim((0, 1))
    plt.xlim((-0.5, len(wps) - 0.5))
    plt.yticks([0, 1], ["0", "1"])
    plt.tight_layout()
    plt.savefig(os.path.join(dname_out, fname_out_fnc(rec_nbr)))
    plt.close("all")
