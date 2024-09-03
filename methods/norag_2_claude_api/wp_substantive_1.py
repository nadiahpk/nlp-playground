# Get Claude to read the first two pages of each Working Paper and
# decide if it should be included for further analysis. We want to
# exclude items like:
#   - drafts of the Final Report of the Antarctic Treaty Consultative Meeting
#   - message of greetings to Antarctic stations
#   - meeting agenda

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, Field
import csv

from my_fncs import load_document

# parameters
# ---

dname_in = "../../results/norag_2_claude_api/texts/"
fnames = [
    "ATCM12_wp002_e_first_two_pages.txt",
    "ATCM12_wp014_rev1_e_first_two_pages.txt",
    "ATCM12_wp015_e_first_two_pages.txt",
    "ATCM12_wp016_rev1_e_first_two_pages.txt",
    "ATCM12_wp017_rev1_e_first_two_pages.txt",
    "ATCM12_wp018_e_first_two_pages.txt",
    "ATCM12_wp019_e_first_two_pages.txt",
    "ATCM12_wp020_e_first_two_pages.txt",
    "ATCM12_wp021_e_first_two_pages.txt",
    "ATCM12_wp022_e_first_two_pages.txt",
    "ATCM12_wp023_e_first_two_pages.txt",
    "ATCM12_wp024_rev1_e_first_two_pages.txt", # draft of final report
    "ATCM12_wp024_rev2_e_first_two_pages.txt", # draft of final report
    "ATCM12_wp025_e_first_two_pages.txt",
    "ATCM12_wp026_e_first_two_pages.txt",
    "ATCM12_wp027_rev1_e_first_two_pages.txt", # greetings
    "ATCM12_wp028_rev1_e_first_two_pages.txt",
]

dname_out = "../../results/norag_2_claude_api/"
fname_out = "wp_substantive.csv"


# ready the structured llm
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
    working_paper_id: str = Field(
        description = "Working Paper ID. Has form 'ANT/{meeting nbr}/{working paper nbr}/[REV{revision nbr}]' with no spaces. Examples: ANT/XII/16/REV1, ANT/XII/19"
    )
    title: str = Field(
        description = "Full title including any parenthetical / subtitle. Capitalisation: title case.",
    )
    is_final_report_draft: bool = Field(
        description = "This is a draft of the Final Report"
    )
    is_substantive: bool = Field(
        description = "This Working Paper contains substantive content contributing to meeting outcomes (e.g., draft Recommendations, discussion papers) rather than procedural elements (e.g., agenda, greeting message to Antarctic stations)"
    )
    justification: str = Field(
        description = "One sentence explaining why Working Paper is or is not substantive."
    )

structured_llm = llm.with_structured_output(WorkingPaperSummary, include_raw=True)

output_dicts = list()
for fname in fnames:
    print(f"Processing file {fname}")
    prompt = load_document(os.path.join(dname_in, fname))

    # make request with raw document as prompt and structured llm
    try:
        output = structured_llm.invoke(prompt)
    except Exception as e:
        print(f" Failed.\n{e}")
        output = None

    # check output from llm and store
    if output:
        output_parsed = output["parsed"]
        if not output_parsed:
            print(" Failed. Raw output follows:")
            print(output["raw"])
        else:
            # print token usage to screen
            usage = output["raw"].usage_metadata
            for token_type, nbr in usage.items():
                print(f"  {token_type} used: {nbr}")

            # add outputs to storage
            output_dicts.append(
                {"file_name": fname} | output["parsed"].dict()
            )

# save outputs to csv
# ---

field_names = [
    "file_name",
    "working_paper_id",
    "title",
    "is_final_report_draft",
    "is_substantive",
    "justification",
]
if output_dicts:
    with open(os.path.join(dname_out, fname_out), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for output_dict in output_dicts:
            writer.writerow(output_dict)
else:
    print("No documents were successfully processed.")
