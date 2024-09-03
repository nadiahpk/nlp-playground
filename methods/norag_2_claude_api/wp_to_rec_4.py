# find links from working papers to recommendations
# 
# -- from wp_to_rec_3.py, added details to prevent Claude from
#    mistaking draft Recommendations contained in the WPs for the
#    final Recommendation
# -- later discovered xml wrappers can also make situation worse, so don't use them


import os
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, Field
import csv

from my_fncs import load_document

# parameters
# ---

# where summaries of documents are saved
dname_in_texts = "../../results/norag_2_claude_api/texts/"
dname_in_sums = "../../results/norag_2_claude_api/"

# input files
fname_wp = "wp_summary_2.csv" # working paper summaries
fname_rec = "rec_summary_1.csv" # recommendation summaries
fname_notcon = "wp_unlikely_recs.csv" # which Recs. each WP almost certainly did not contribute to

# where to save things
dname_out = dname_in_sums
fname_out = "wp_to_rec_4.csv"

# use the data files to extract basic info about WPs and Recs.
# ---

df_wp = pd.read_csv(os.path.join(dname_in_sums, fname_wp), index_col = "working_paper_id")
df_rec = pd.read_csv(os.path.join(dname_in_sums, fname_rec), index_col = "recommendation_id")
df_contrib = pd.read_csv(os.path.join(dname_in_sums, fname_notcon), index_col = "working_paper_id")

# set of all working papers and recs
wps = list(df_wp.index)
recs = list(df_rec.index)

# for each working paper, set of recommendations it might have contributed to
maybe_contribsD = {
    wp: [R for R in recs if R not in string_of_Rs.split(" | ")]
    for wp, string_of_Rs in zip(df_contrib.index, df_contrib["did_not_contrib_to"])
}


# for each potential (WP, Rec) pair, ask if WP contributed to Rec.
# ---

# ready the structured llm

# instantiate
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0,
    # max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# structured llm with request schema 
class Request(BaseModel):
    working_paper_contains_draft: bool = Field(
        description="The Working Paper contains a Draft Recommendation"
    )
    contributed: bool = Field(
        description="The Working Paper (and any Draft Recommendations it may contain) contributed to the final Recommendation."
    )
    justification: str = Field(
        description="One-paragraph explanation of contributions."
    )

structured_llm = llm.with_structured_output(Request, include_raw=True)


# for each (working paper, recommendation) pair, ask if working apper contributed
# ---

# storage of output dictionaries
output_dicts = list()

# for wp in wps:
for wp in wps:
    print(f"Processing Working Paper {wp} ...")
    maybe_contribs = maybe_contribsD[wp]

    if maybe_contribs:
        # text of working paper
        wp_text = load_document(os.path.join(dname_in_texts, df_wp.loc[wp]["file_name"]))

    # for rec in maybe_contribs:
    for rec in maybe_contribs:
        print(f" Asking about connection to Recommendation {rec} ...")
        rec_text = load_document(os.path.join(dname_in_texts, df_rec.loc[rec]["file_name"]))

        prompt = (
            "Two separate documents follow that may or may not be related. \n\n\n"
            + "\n\n\n========== START OF DOCUMENT 1: WORKING PAPER ==========\n\n\n"
            + wp_text
            + "\n\n\n========== END OF DOCUMENT 1 (WORKING PAPER) ==========\n\n\n"
            "\n\n\n========== START OF DOCUMENT 2: FINAL RECOMMENDATION ==========\n\n\n"
            + rec_text
            + "\n\n\n========== END OF DOCUMENT 2 (FINAL RECOMMENDATION) ==========\n\n\n"
        )
        
        # make request with xml-wrapped prompt and structured llm
        try:
            output = structured_llm.invoke(prompt)
        except Exception as e:
            print(f" Failed.\n{e}")
            output = None
            continue

        # check output from llm and store
        output_parsed = output["parsed"]
        if output_parsed is None:
            print(" Failed. Raw output follows:")
            print(output["raw"])
            continue

        # print token usage to screen
        usage = output["raw"].usage_metadata
        for token_type, nbr in usage.items():
            print(f"  {token_type} used: {nbr}")

        # add outputs to storage
        output_dicts.append(
            {"working_paper_id": wp, "recommendation_id": rec} | output["parsed"].dict()
        )


# write to file
# ---

field_names = [
    "working_paper_id", 
    "recommendation_id",
    "contributed",
    "justification"
]
with open(os.path.join(dname_out, fname_out), "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=field_names, quoting=csv.QUOTE_ALL, extrasaction="ignore"
    )
    writer.writeheader()
    for output_dict in output_dicts:
        writer.writerow(output_dict)