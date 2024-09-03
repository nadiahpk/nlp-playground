# use wp summaries and agenda summaries to identify which working papers were
# discussed under which agenda items
# -- I don't know if I really believe the results of this
# -- didn't do the output safely

import os
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, Field
import csv

from my_fncs import documents_to_xml_string

# parameters
# ---

# where input data is saved
dname_in = "../../results/norag_2_claude_api/"
fname_agenda = "agenda_summary_2.csv"
fname_wp = "wp_summary_2.csv"  # summary of working papers

# where to save output
dname_out = dname_in
fname_out = "wp_to_agenda_1.csv"


# ready the llm
# ---

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0,
    # max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

class Request(BaseModel):
    agenda_item_numbers: str = Field(
        description="Pipe-separated Agenda Item number(s) where Working Paper was discussed (e.g., '10 | 11')"
    )
    justification: str = Field(
        description="Evidence that this Working Paper was discussed"
    )

structured_llm = llm.with_structured_output(Request, include_raw=True)

# read in summaries and prepare for llm
# ---

df_wp = pd.read_csv(os.path.join(dname_in, fname_wp), index_col="working_paper_id")
keep_cols = [
    "full_title",
    "submitted_by",
    "agenda_item_referenced",
    "agenda_item_numbers",
    "agenda_item_title",
    "nbr_of_articles_referenced",
    "which_articles",
    "nbr_of_recommendations_referenced",
    "which_recommendations",
    "summary",
]
df_wp = df_wp[keep_cols]

# columns of working-paper summary to show llm
llm_cols = [
    "full_title",
    "submitted_by",
    "nbr_of_articles_referenced",
    "which_articles",
    "nbr_of_recommendations_referenced",
    "which_recommendations",
    "summary",
]

# subset to substantive agenda and keep selected columns
df_agenda = pd.read_csv(os.path.join(dname_in, fname_agenda))
df_agenda = df_agenda[df_agenda["is_substantive"]]
keep_cols = [
    "agenda_item_nbr",
    "title",
    "is_substantive",
    "nbr_articles_referenced",
    "which_articles",
    "nbr_recommendations_referenced",
    "which_recommendations",
    "summary",
]
df_agenda = df_agenda[keep_cols]
agenda_csv = df_agenda.to_csv(quoting=csv.QUOTE_ALL)


# for each working paper, identify which agenda item
# ---

output_dicts = list()
for wp in df_wp.index:
    print(f"Processing {wp}...")
    if df_wp.loc[wp]["agenda_item_referenced"]:
        # we know the agenda item from the wp itself
        output_dict = {
            "working_paper_id": wp,
            "agenda_item_numbers": df_wp.loc[wp]["agenda_item_numbers"],
            "justification": "WP cites Agenda Item number"
        }
    else:
        # turn the information about this working into plain text
        wp_texts = [
            f"{item_name}: {item_info}"
            for item_name, item_info in df_wp.loc[wp][llm_cols].items()
        ]
        wp_text = "\n\n".join(wp_texts)

        # create the xml-wrapped prompt with agenda and WP summaries
        document_dictionary = {
        "agenda_items_summary.csv": agenda_csv,
        "working_paper_summary.csv": wp_text
        }
        prompt = documents_to_xml_string(document_dictionary)

        # get output
        output = structured_llm.invoke(prompt)
        output_dict = {
            "working_paper_id": wp,
            "agenda_item_numbers": output["parsed"].agenda_item_numbers,
            "justification": output["parsed"].justification
        }
    
    # store
    output_dicts.append(output_dict)

# write to file
# ---

field_names = ["working_paper_id", "agenda_item_numbers", "justification"]
with open(os.path.join(dname_out, fname_out), "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL, extrasaction="ignore")
    writer.writeheader()
    for output_dict in output_dicts:
        writer.writerow(output_dict)
