# goal: use long context and schema to identify which WPs almost certainly did not contribute to which Recs
#
# -- can probably skip file name (NOTE below)
# --

import pandas as pd
import numpy as np
import csv
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import os
from langchain_anthropic import ChatAnthropic

from my_fncs import load_document, documents_to_xml_string

# parameters
# ---

# where documents are stored
dname_in = "../../results/norag_2_claude_api/"
fname_recs = "rec_summary_1.csv"
fname_wps = "wp_summary_2.csv"

# where to save documents
dname_out = dname_in
fname_out = "wp_unlikely_recs.csv"


# create structured LLM
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

class WorkingPaper(BaseModel):
    working_paper_id: str = Field(description="Working Paper ID")
    did_not_contrib_to: str = Field(
        description = "List of Recommendations to which this Working Paper almost certainly did NOT contribute",
        format = "Pipe-separated list of Recommendation IDs, e.g., 'XII-1 | XII-2'"
    )

class Request(BaseModel):
    response: List[WorkingPaper] = Field(
      description = "List of every working paper")

# create the structured llm with schema
structured_llm = llm.with_structured_output(Request, include_raw=True)


# create the xml-wrapped prompt
# ---

# create trimmed recommendations.csv
df_recs_orig = pd.read_csv(os.path.join(dname_in, fname_recs))
keep_cols = [
    "recommendation_id",
    "full_title",
    # "which_articles",
    # "which_recommendations",
    "summary",
]
df_recs = df_recs_orig[keep_cols]
"""
df_recs.columns = [
    "recommendation_id",
    "full_title",
    # "articles_referenced",
    # "recommendations_referenced",
    "summary",
]
"""
text_recs = df_recs.to_csv(index=False, quoting=csv.QUOTE_ALL)

# create trimmed working_papers.csv
df_wps_orig = pd.read_csv(os.path.join(dname_in, fname_wps))
keep_cols = [
    "working_paper_id",
    "full_title",
    # "which_articles",
    # "which_recommendations",
    "summary",
]
df_wps = df_wps_orig[keep_cols]
"""
df_wps.columns = [
    "working_paper_id",
    "full_title",
    "articles_referenced",
    "recommendations_referenced",
    "summary",
]
"""

# split up if longer than 10
len_wps = len(df_wps)
if len_wps <= 10:
  text_wpsV = [df_wps.to_csv(index=False, quoting=csv.QUOTE_ALL)]
else:
  # split into csvs of approx 10 rows
  nbr_divs = int(np.ceil(len_wps / 10))
  div_sz = int(np.ceil(len_wps / nbr_divs))
  divs = np.arange(0, len_wps, div_sz)

  # store splits into list
  text_wpsV = list()
  for i in range(len(divs)-1):
    text_wpsV.append(df_wps.iloc[divs[i]:divs[i+1]].to_csv(index=False, quoting=csv.QUOTE_ALL))
  text_wpsV.append(df_wps.iloc[divs[-1]:].to_csv(index=False, quoting=csv.QUOTE_ALL))


# for each subset of Working Papers, process the prompt using a structured schema
# ---

output_dicts = list()
for i, text_wps in enumerate(text_wpsV):
  print(f"Processing group {i} of {len(text_wpsV)}")

  # wrap the documents in xml tags
  document_dictionary = {
    "working_papers.csv": text_wps,
    "recommendations.csv": text_recs,
  }
  prompt = documents_to_xml_string(document_dictionary)

  # process the prompt using the structured llm
  try:
    output = structured_llm.invoke(prompt)
  except Exception as e:
      print(f" Failed.\n{e}")
      output = None

  if output is None:
    continue

  output_parsed = output["parsed"]
  if output_parsed is None:
    print(" Failed. Raw output follows:")
    print(output["raw"])
    continue

  # print token usage to screen
  usage = output["raw"].usage_metadata
  for token_type, nbr in usage.items():
      print(f"  {token_type} used: {nbr}")

  # extract list of dicts with structure: 
  output_dicts += output_parsed.dict()["response"]

# append file names of working paper to each dictionary
# ---

# NOTE - you can probably skip this
for output_dict in output_dicts:
  wp = output_dict["working_paper_id"]
  output_dict["file_name"] = df_wps_orig[df_wps_orig["working_paper_id"] == wp]["file_name"].iloc[0]

# write information to file
# ---

field_names = ["working_paper_id", "file_name", "did_not_contrib_to"]
with open(os.path.join(dname_out, fname_out), "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for output_dict in output_dicts:
        writer.writerow(output_dict)