import pandas as pd
import os
import graphviz


# parameters
# ---

this_meet = "XII"
next_meet = "XIII"

# where data is
dname_in = "../../results/norag_2_claude_api/"
fname_wp = "working_paper_connections_2.csv"
fname_ai = "agenda_item_connections_2.csv"

# where to save
dname_out = dname_in
fname_out = "connections_3"

# read in
df_wp = pd.read_csv(os.path.join(dname_in, fname_wp), index_col=0)
df_ai = pd.read_csv(os.path.join(dname_in, fname_ai), index_col=0)


# create structures for dottifying
# ---

# working papers contribute to recommendations, wp -> rec
wp2recs = {
    wp: (
        set(df_wp.loc[wp]["rec_ids"].split(" | ")) if type(df_wp.loc[wp]["rec_ids"]) is str
        else set()
    )
    for wp in df_wp.index
}

# each agenda item ai is a cluster of working papers wp
ai2wps = {
    ai: set(df_ai.loc[ai]["wp_ids"].split(" | "))
    for ai in df_ai.index if type(df_ai.loc[ai]["wp_ids"]) is str
}

# some agenda items contribute to recommendations independently of the working papers
# they contain, so created ai -> rec that doesn't include -> rec 
# already covered by working papers in agenda

ai2recs_already = {ai: set() for ai in df_ai.index}
for ai, wps in ai2wps.items():
    for wp in wps:
        ai2recs_already[ai] = ai2recs_already[ai].union(wp2recs[wp])

ai2recs = {
    ai: set(df_ai.loc[ai]["rec_ids"].split(" | ")).difference(ai2recs_already[ai])
    for ai, recs in zip(df_ai.index, df_ai["rec_ids"]) if type(df_ai.loc[ai]["rec_ids"]) is str
}
ai2recs = {k: v for k, v in ai2recs.items() if v} # remove empty-valued items


# connections to past and future meetings
# ---

# working paper to other meetings
wp2next = [wp for wp, flag in zip(df_wp.index, df_wp["next_meet"]) if flag]
past2wp = [wp for wp, flag in zip(df_wp.index, df_wp["past_meets"]) if flag]

ai2next = [ai for ai, flag in zip(df_ai.index, df_ai["next_meet"]) if flag]
past2ai = [ai for ai, flag in zip(df_ai.index, df_ai["past_meets"]) if flag]


# put all dictionaries into a super-dictionary of this meeting
# ---

aiDD = dict()
for ai, wps in ai2wps.items():
    if ai not in aiDD:
        aiDD[ai] = dict()
    aiDD[ai]["wps"] = wps

for ai, recs in ai2recs.items():
    if ai not in aiDD:
        aiDD[ai] = dict()
    aiDD[ai]["to_recs"] = recs

for ai in ai2next:
    if ai not in aiDD:
        aiDD[ai] = dict()
    if "to_futures" not in aiDD[ai]:
        aiDD[ai]["to_future"] = list()
    aiDD[ai]["to_future"].append(next_meet)

for ai in past2ai:
    if ai not in aiDD:
        aiDD[ai] = dict()

wpDD = dict()
for wp, recs in wp2recs.items():
    if wp not in wpDD:
        wpDD[wp] = dict()
    wpDD[wp]["to_recs"] = recs

for wp in wp2next:
    if wp not in wpDD:
        wpDD[wp] = dict()
    if "to_futures" not in wpDD[wp]:
        wpDD[wp]["to_future"] = list()
    wpDD[wp]["to_future"].append(next_meet)

meetingsD = {
    "past":
    {
        "meet_id": "past",
        "aiDD": dict(), # "aiDD": { "0": {"wps": {"ANT/XI/0"}} },
        "wpDD": dict(),
        "unknown": list(),
    },
    this_meet:
    {
        "meet_id": this_meet,
        "aiDD": aiDD,
        "wpDD": wpDD,
        "unknown": list(),
    },
    next_meet:
    {
        "meet_id": next_meet,
        "aiDD": dict(),
        "wpDD": dict(),
        "unknown": list(),
    },
    # "future":
    # {
        # "meet_id": "future",
        # "aiDD": dict(),
        # "wpDD": dict(),
        # "unknown": list(),
    # },
}
meetings = list(meetingsD.keys())

# create dot file
# ---

# Assuming ai2wps, wp2recs, and ai2recs are defined as in your example
# dot_graph = create_dot_graph(ai2wps, wp2recs, ai2recs)
if True:

    dot_graph = graphviz.Digraph(comment='Agenda Items, Working Papers, and Recommendations')
    dot_graph.attr(rankdir='TB', compound="True", newrank="True")

    # create clusters for meetings
    for meet_id in meetingsD.keys():
        meeting = meetingsD[meet_id]
        aiDD = meeting["aiDD"]
        wpDD = meeting["wpDD"]

        # meeting source and sink nodes
        sink = f"sink_{meet_id}"
        dot_graph.node(sink, shape="point", label="", style="invis")

        # meeting -> ai
        # ---

        dot_graph.node(meet_id, label=f"ATCM {meet_id}", fontsize="24", shape="box", color="white")
        for ai in aiDD:
            ai_name = f"agenda_{meet_id}-{ai}"
            dot_graph.node(ai_name, label = f"Agenda Item {ai.replace(' | ', ', ')}", fontsize="20", shape="box", color="#dddddd")
            dot_graph.edge(meet_id, ai_name, style="invis")

        # put working papers on the same rank
        # ---

        c = graphviz.Digraph(graph_attr={"rank": "same"})
        # include this later to put labels on lhs
        # c.node(meet_id + "_label", f"ATCM {meet_id}", fontsize="40", shape="plain")
        for ai, aiD in aiDD.items():
            # add all working papers
            if "wps" in aiD:
                for wp in aiD["wps"]:
                    c.node(wp, f"WP{wp.split('/')[2]}", shape="box", fillcolor='#bbccee', style="rounded, filled")
                    # ac.edge(fake, wp, style="invis")
            else:
                # create a fake working paper
                c.node(ai + "fake_paper", "", shape="point", style="invis")
        dot_graph.subgraph(c)

        # link each agenda item to its wps
        # ---
        for ai, aiD in aiDD.items():
            ai_name = f"agenda_{meet_id}-{ai}"

            # edge to each working papers
            if "wps" in aiD:
                for wp in aiD["wps"]:
                    dot_graph.edge(ai_name, wp, dir="none", style="invis")
            else:
                dot_graph.edge(ai_name, ai + "fake_paper", dir="none", style="invis")

            # link fake to source
            # dot_graph.edge(source, ai_name, style="invis")


        # Add all recommendation nodes
        # ---

        all_recs = set()
        for wpD in wpDD.values():
            if "to_recs" in wpD:
                all_recs.update(wpD["to_recs"])
        for aiD in aiDD.values():
            if "to_recs" in aiD:
                all_recs.update(wpD["to_recs"])
        for rec in all_recs:
            dot_graph.node(rec, f"R{rec.split('-')[1]}", shape='box', style="filled", fillcolor="#ccddaa")

        # rec -> sink
        for rec in all_recs:
            dot_graph.edge(rec, sink, style="invis")

        # wp, ai -> rec
        # ---

        for wp, wpD in wpDD.items():
            if "to_recs" in wpD:
                for rec in wpD["to_recs"]:
                    dot_graph.edge(wp, rec)

        for ai, aiD in aiDD.items():
            if "to_recs" in aiD:
                for rec in aiD["to_recs"]:
                    dot_graph.edge(f"agenda_{meet_id}-{ai}", rec, ltail=f'cluster_agenda_{meet_id}-{ai}')
                    # dot_graph.edge(f"agenda_{meet_id}-{ai}", rec)


        # wp, ai -> next
        # ---

        for wp, wpD in wpDD.items():
            if "to_future" in wpD:
                for future_meet in wpD["to_future"]:
                    dot_graph.edge(wp, future_meet)

        for ai, aiD in aiDD.items():
            if "to_future" in aiD:
                ai_name = f"agenda_{meet_id}-{ai}"
                for future_meet in aiD["to_future"]:
                    dot_graph.edge(ai_name, future_meet, ltail=f'cluster_agenda_{meet_id}-{ai}')

        # try putting wps in their own subgraphs?
        # ---

        for ai, aiD in aiDD.items():
            ac = graphviz.Digraph(name=f'cluster_agenda_{meet_id}-{ai}')
            ac.attr(style='filled')
            ac.attr(color="#dddddd")
            ai_name = f"agenda_{meet_id}-{ai}"
            ac.node(ai_name)
            # add all working papers
            if "wps" in aiD:
                for wp in aiD["wps"]:
                    ac.node(wp)
                    # ac.edge(fake, wp, style="invis")
            else:
                # create a fake working paper
                ac.node(ai + "fake_paper")
            dot_graph.subgraph(ac)

        # append the subgraph
        # dot_graph.subgraph(mc)

    # link meetings in order
    # ---

    # link source-sink in order
    for i in range(len(meetings)-1):
        id1 = meetings[i]
        id2 = meetings[i+1]
        sink = f"sink_{id1}"
        dot_graph.edge(sink, id2, style="invis")

    # Include this later to put labels on lhs
    """
    # link meeting labels in order
    for i in range(len(meetings)-1):
        id1 = meetings[i]
        id2 = meetings[i+1]
        dot_graph.edge(id1 + "_label", id2 + "_label")
    """

    # need to hack a link from past to its sink
    dot_graph.edge("past", "sink_past", style="invis")

    # link current meeting to past
    # ---

    for ai in past2ai:
        ai_name = f"agenda_{this_meet}-{ai}"
        dot_graph.edge("past", ai_name, lhead=f'cluster_agenda_{this_meet}-{ai}')

    for wp in past2wp:
        dot_graph.edge("past", wp)


    # return dot_graph

# Save the graph as a .dot file
# ---

dot_graph.render(os.path.join(dname_out, fname_out), format='dot', cleanup=True)
print(f"DOT file has been generated: {fname_out}.dot")