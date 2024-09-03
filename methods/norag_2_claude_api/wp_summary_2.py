# rewrite the working paper summariser using what I've learnt since ../norag_1_claude_api/
# -- was having real failures with Dictionaries and other structures,
#    so revert to something simpler like what worked in ../norag_1_claude_api/

import pandas as pd
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, Field
import csv

from my_fncs import load_document

# parameters
# ---

# next meeting's number in Roman numerals (provides a hint for llm)
next_meet = "XIII (Thirteenth)"

# where wps are stored
dname_in = "../../results/norag_2_claude_api/texts/"

# summarises first few pages
fname_wp_sum = "../../results/norag_2_claude_api/wp_substantive.csv"

# where to save the working paper summaries
dname_out = "../../results/norag_2_claude_api/"
fname_out = "wp_summary_2.csv"


# get list of files of substantive working papers
# ---

df = pd.read_csv(fname_wp_sum)
df = df[df["is_substantive"]] # subset
fnames = [
    fname_first_pages.replace("_first_two_pages", "")
    for fname_first_pages in df["file_name"]
]


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
    working_paper_id: str = Field(
        description = "Working Paper ID. Has form 'ANT/{meeting nbr}/{working paper nbr}/[REV{revision nbr}]' with no spaces. Examples: ANT/XII/16/REV1, ANT/XII/19"
    )
    full_title: str = Field(
        description = "Full title including any parenthetical / subtitle",
        capitalisation = "Title Case"
    )
    submitted_by: str = Field(
        description = "Pipe-separated list of countries / organisations (e.g., 'Australia | Argentina')"
    )
    submitted_by_country: bool = Field(
        description = "At least one submitter was a country"
    )

    # agenda links
    agenda_item_referenced: bool
    agenda_item_numbers: str = Field(
        description = "Pipe-separated list of Agenda Item numbers; or 'None'"
    )
    agenda_item_title: str = Field(
        description = "Agenda Item title; or 'None'",
        capitalisation = "Title case"
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

    previous_meeting_referenced: bool = Field(
        description="Describes unresolved matters deferred from a previous meeting"
    )
    which_previous_meetings: str = Field(
        description="Pipe-separated list of meetings; or 'None'"
    )
    previous_meetings_summary: str = Field(
        description="Summary of discussion at each previous meeting (one sentence each); or 'None'"
    )
    """
    previous_meeting_referenced: bool = Field(
        description="Describes discussions of matters unresolved or deferred at a previous meeting"
    )
    which_previous_meetings: str = Field(
        description="Pipe-separated list of meetings; or 'None'"
    )
    previous_meetings_summary: str = Field(
        description="Summary of discussion at each previous meeting (one sentence each); or 'None'"
    )
    """

    # link to other documents
    nbr_of_previous_papers_referenced: str = Field(
        description="Count previous working documents, such as Working Papers, discussion papers, or draft recommendations"
    )
    which_previous_papers: str = Field(
        description="Pipe-separated list of all papers; or 'None'"
    )
    previous_papers_summary: str = Field(
        description="Summary of each previous paper; or 'None'"
    )

    # postpone or discuss further at a future meeting
    future_meeting_referenced: bool = Field(
        description="Mentions that matters will be further discussed at a future meeting"
    )
    future_meetings_includes_next: bool = Field(
        description=f"Future meetings mentioned includes Consultative Meeting {next_meet}."
    )
    which_future_meetings: str = Field(
        description="Pipe-separated list of meetings; or 'None'"
    )
    future_meetings_summary: str = Field(
        description="Summary of proposal to discuss at future meetings; or 'None'"
    )

    # content
    contains_draft_recommendation: bool = Field(
        description="Contains a draft Recommendation"
    )
    summary: str = Field(description="One-paragraph plain-English summary of Working Paper")

structured_llm = llm.with_structured_output(WorkingPaperSummary, include_raw=True)

# get a summary of each substantive working paper
# ---

output_dicts = list()
for fname_in in fnames[:12]:

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
    'working_paper_id',
    'full_title',
    'submitted_by',
    'submitted_by_country',
    'agenda_item_referenced',
    'agenda_item_numbers',
    'agenda_item_title',
    'nbr_of_articles_referenced',
    'which_articles',
    'nbr_of_recommendations_referenced',
    'which_recommendations',
    'previous_meeting_referenced',
    'which_previous_meetings',
    'previous_meetings_summary',
    'nbr_of_previous_papers_referenced',
    'which_previous_papers',
    'previous_papers_summary',
    'future_meeting_referenced',
    'future_meetings_includes_next',
    'which_future_meetings',
    'future_meetings_summary',
    'contains_draft_recommendation',
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