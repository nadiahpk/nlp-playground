# rewrite the Recommendations summariser

import pandas as pd
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, Field
import csv

from my_fncs import load_document

# parameters
# ---

this_meeting = "XII"

# where recs are stored
dname_in = "../../results/norag_2_claude_api/texts/"

fnames = [
    "ATCM12_R1.txt",
    "ATCM12_R2.txt",
    "ATCM12_R3.txt",
    "ATCM12_R4.txt",
    "ATCM12_R5.txt",
    "ATCM12_R6.txt",
    "ATCM12_R7.txt",
    "ATCM12_R8.txt",
]

# next meeting's number in Roman numerals (provides a hint for llm)
# next_meet = "XIII (Thirteenth)"

# where to save the Recommendation summaries
dname_out = "../../results/norag_2_claude_api/"
fname_out = "rec_summary_1.csv"

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

# structured llm with request schema 
class WorkingPaperSummary(BaseModel):
    # basic info
    recommendation_id: str = Field(
        description = f"Recommendation ID (e.g., {this_meeting}-1, {this_meeting}-2)"
    )
    full_title: str = Field(
        description = "Full title including any parenthetical / subtitle",
        capitalisation = "Title Case"
    )

    # references to Articles and Recommendations
    nbr_of_articles_referenced: int
    which_articles: str = Field(
        description="Pipe-separated list of Article numbers (Roman numerals); or 'None'"
    )
    nbr_of_recommendations_referenced: int
    which_recommendations: str = Field(
        description="Pipe-separated list of Recommendation numbers (e.g., 'XI-1 | III-I'); or 'None'"
    )
    summary: str = Field(description="One-paragraph plain-English summary of Recommendation")

structured_llm = llm.with_structured_output(WorkingPaperSummary, include_raw=True)


# get a summary of each Recommendation
# ---

output_dicts = list()
for fname_in in fnames:

    # create a prompt from the relevant file
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
    if not output_parsed:
        print(f" {fname_in} failed. Parsing error follows:")
        print(output["parsing_error"])
        print("Raw output follows:")
        print(output["raw"])
        continue

    # print token usage to screen
    usage = output["raw"].usage_metadata
    for token_type, nbr in usage.items():
        print(f"  {token_type} used: {nbr}")

    # add the file name
    output_dict = output_parsed.dict()
    output_dict["file_name"] = fname_in
    
    # store this output
    output_dicts.append(output_dict)


# save outputs to csv
# ---

field_names = [
    'file_name',
    'recommendation_id',
    'full_title',
    'nbr_of_articles_referenced',
    'which_articles',
    'nbr_of_recommendations_referenced',
    'which_recommendations',
    'summary', 
]


if output_dicts:
    with open(os.path.join(dname_out, fname_out), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for output_dict in output_dicts:
            writer.writerow(output_dict)
else:
    print("No documents were successfully processed.")
