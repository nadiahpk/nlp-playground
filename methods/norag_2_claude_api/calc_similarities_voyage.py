# Calculate the cosine similarity between each Working Paper and Recommendation in 1983,
# and write it to a csv for later analysis

import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
import pandas as pd
import voyageai

# ===========================================

# inputs
# ---

# where information is about working papers and recommendations
dname_in = "../../results/norag_2_claude_api/"
dname_in_texts = "../../results/norag_2_claude_api/texts/"
fname_wp = "wp_summary_2.csv" # working paper summaries
fname_rec = "rec_summary_1.csv" # recommendation summaries
fname_wp2rec = "wp_to_rec_4.csv" # WP to Recommendations from LLM

dname_out = dname_in
fname_out = "similarities_voyage_law_2.csv"


# read in documents
# ---

df = pd.read_csv(os.path.join(dname_in, fname_wp))
wps = df["working_paper_id"]
wp_documents = [
    Path(os.path.join(dname_in_texts, fname)).read_text()
    for fname in df["file_name"]
]

df = pd.read_csv(os.path.join(dname_in, fname_rec))
recs = df["recommendation_id"]
rec_documents = [
    Path(os.path.join(dname_in_texts, fname)).read_text()
    for fname in df["file_name"]
]


"""
# get contributions to Recommendations of WPs NOTE - move this
# ---

df = pd.read_csv(os.path.join(dname_in, fname_wp2rec))
df = df[df["contributed"]]
recs = set(df["recommendation_id"])
rec2wps = {
    rec: set(df[df["recommendation_id"] == rec]["working_paper_id"])
    for rec in recs
}
"""


# calculate embeddings
# ---

vo = voyageai.Client() # automatically use environment var VOYAGE_API_KEY

wp_embeddings = vo.embed(
    wp_documents, model="voyage-law-2", input_type="document"
).embeddings

rec_embeddings = vo.embed(
    rec_documents, model="voyage-law-2", input_type="document"
).embeddings

# cosine similarity between them
# TODO dot product is also possible here
similaritys = cosine_similarity(wp_embeddings, rec_embeddings)

# save to csv
df = pd.DataFrame(similaritys, columns = recs, index = wps)
df.to_csv(os.path.join(dname_out, fname_out), index=True)
