# use the agenda chunks to describe each agenda item
# 
# -- Does a much better job than agenda_summary_1.py picking up papers referenced
#
# Things I learnt later:
# -- I'm using List[str] here, which misbehaves. See wp_summary_1.py for remedy.
# -- Dictionary also misbehaves. Don't use structures in future.
# -- Order of structure makes a difference! For example, I'm asking 
#    about Recommendations after previous meetings, but it would be
#    better to to put it before.

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict
import csv

from my_fncs import load_document

# parameters
# ---

# next meeting's number in Roman numerals (provides a hint for llm)
next_meet = "(Thirteenth, XIII)"

# where input data is saved
dname_in = "../../results/norag_2_claude_api/texts/"
chunk_max = 11 # chunk names range from 0 to chunk_max
fname_in_fnc = lambda chunk_id: f"ATCM12_fr001_e_minutes_chunk_{chunk_id}.txt"  # noqa

# where to save output
dname_out = "../../results/norag_2_claude_api/"
fname_out = "agenda_summary_2.csv"


# ready the llm
# ---

# instantiate
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0,
    # max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


class AgendaItemSummary(BaseModel):
    # basic agenda-item info
    agenda_item_nbr: List[int] = Field(description="List of Agenda Item numbers")
    title: str = Field(description="Agenda Item title. Capitalisation: title case")
    start_par: int = Field(
        description="Starting paragraph number. (Ignore page numbers)."
    )
    end_par: int = Field(description="Ending paragraph number. (Ignore page numbers).")

    # postpone or discuss further at a future meeting
    future_meetings_referenced: bool = Field(
        description="Mentions that matters will be further discussed at a future meeting"
    )
    future_meetings_includes_next: bool = Field(
        description=f"Future meetings mentioned includes next Consultative Meeting {next_meet}."
    )
    future_meetings_summary: Optional[Dict[str, str]] = Field(
        description="Summary of matters for future meeting",
        key_description="Name of future meeting",
        value_description="Summary of matter",
    )

    # link to past discussions, meetings
    past_discussions_referenced: bool = Field(description="Mentions past discussions at previous meetings")
    past_discussions_summary: Optional[Dict[str, str]] = Field(
        description="Summary of past discussions", 
        key_description="Name of meeting where discussed",
        value_description="Summary of discussion",
    )

    # link to other documents
    papers_referenced: bool = Field(
        description="Mentions a previous working document, such as a Working Paper, discussion paper, or draft recommendation"
    )
    papers_summary: Optional[Dict[str, str]] = Field(
        description="Summary of relevance of each paper",
        key_description="Name of paper",
        value_description="Summary of paper and how it relates to the agenda item",
    )
    withdrawn_recommendations: bool = Field(
        description="Mentions draft Recommendations being withdrawn"
    )
    withdrawns_summary: Optional[Dict[str, str]] = Field(
        description="Summary of each withdrawn Recommendation",
        key_description="Name of Recommendation",
        value_description="Summary of Recommendation and reason for withdrawal",
    )

    # references to Articles and Recommendations
    nbr_articles_referenced: int = Field(description="Number of Articles mentioned")
    which_articles: Optional[List[str]] = Field(
        description="Article numbers (Roman numerals)"
    )
    nbr_recommendations_referenced: int = Field(
        description="Number of Recommendations mentioned"
    )
    which_recommendations: Optional[List[str]] = Field(
        description="Recommendations numbers (e.g., ['XI-1', 'III-I'])"
    )

    # overall summary
    summary: str = Field(description="One-paragraph summary of Agenda Item")
    is_substantive: bool = Field(
        description="Agenda Item concerns substantive matters (e.g., policy, recommendations) as opposed procedural items (e.g., opening remarks, accepting the agenda)"
    )


structured_llm = llm.with_structured_output(AgendaItemSummary, include_raw=True)

output_dicts = list()
for chunk_id in range(chunk_max+1):
    # create a prompt from the relevant chunk file
    fname_in = fname_in_fnc(chunk_id)
    prompt = load_document(os.path.join(dname_in, fname_in))

    # process chunk with structured llm
    print(f"Processing {fname_in} ...")
    try:
        output = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"For agenda file {fname_in}, structured_llm.invoke() failed.\n{e}")
        output = None
        continue

    # check parsed output
    output_parsed = output["parsed"]
    if output_parsed is None:
        print(f" {fname_in} failed. Raw output follows:")
        print(output["raw"])
        continue

    # print token usage to screen
    usage = output["raw"].usage_metadata
    for token_type, nbr in usage.items():
        print(f"  {token_type} used: {nbr}")

    # convert all lists to double-pipe-separated strings (" || ")
    # and dictionaries to pipe-separated "key | value" strings
    output_dict = dict()
    for key, blob in output_parsed.dict().items():
        if type(blob) is list:
            new_blob = " | ".join(str(v) for v in blob)
        elif type(blob) is dict:
            new_blob = " || ".join(" | ".join([str(key), str(value)]) for key, value in blob.items())
        else:
            new_blob = blob
        output_dict[key] = new_blob

    # add the file name of the chunk
    output_dict["file_name"] = fname_in
    
    # store this output
    output_dicts.append(output_dict)


# save outputs to csv
# ---

field_names = [
    "file_name",
    "agenda_item_nbr",
    "title",
    "start_par",
    "end_par",
    "future_meetings_referenced",
    "future_meetings_includes_next",
    "future_meetings_summary",
    "past_discussions_referenced",
    "past_discussions_summary",
    "papers_referenced",
    "papers_summary",
    "withdrawn_recommendations",
    "withdrawns_summary",
    "nbr_articles_referenced",
    "which_articles",
    "nbr_recommendations_referenced",
    "which_recommendations",
    "summary",
    "is_substantive"
]


if output_dicts:
    with open(os.path.join(dname_out, fname_out), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for output_dict in output_dicts:
            writer.writerow(output_dict)
else:
    print("No documents were successfully processed.")