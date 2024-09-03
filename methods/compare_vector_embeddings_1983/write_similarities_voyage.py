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

# where to write results
fname_out_fnc = lambda model_name: f"../../results/compare_vector_embeddings_1983/similarities_voyage_{model_name}.csv" # noqa

# models to try
model_names = ["voyage-2", "voyage-law-2"]

# number of working papers and recommendations
nbr_wps = 20
nbr_recs = 8

# hardcoded for now, the file names of WPs and Recommendations in order

dname_wp = "../../data_processed/documents/1983/"
fname_wps = [
    "ATCM12_wp001_rev1_e_ocr_cleaned.txt",
    "ATCM12_wp002_e_ocr_cleaned.txt",
    "ATCM12_wp003_e_ocr_cleaned.txt",
    "ATCM12_wp004_rev1_e_ocr_cleaned.txt",
    "ATCM12_wp005_e_ocr_cleaned.txt",
    "ATCM12_wp006_e_ocr_cleaned.txt",
    "ATCM12_wp007_e_ocr_cleaned.txt",
    "ATCM12_wp008_e_ocr_cleaned.txt",
    "ATCM12_wp009_e_ocr_cleaned.txt",
    "ATCM12_wp010_e_ocr_cleaned.txt",
    "ATCM12_wp011_e_ocr_cleaned.txt",
    "ATCM12_wp012_e_ocr_cleaned.txt",
    "ATCM12_wp013_e_ocr_cleaned.txt",
    "ATCM12_wp014_rev1_e_ocr_cleaned.txt",
    "ATCM12_wp015_e_ocr_cleaned.txt",
    "ATCM12_wp016_rev1_e_ocr_cleaned.txt",
    "ATCM12_wp017_rev1_e_ocr_cleaned.txt",
    "ATCM12_wp018_e_ocr_cleaned.txt",
    "ATCM12_wp019_e_ocr_cleaned.txt",
    "ATCM12_wp020_e_ocr_cleaned.txt",
]

dname_rec = "../../data_processed/documents/1983/"
fname_recs = [
    "ATCM12_R1.txt",
    "ATCM12_R2.txt",
    "ATCM12_R3.txt",
    "ATCM12_R4.txt",
    "ATCM12_R5.txt",
    "ATCM12_R6.txt",
    "ATCM12_R7.txt",
    "ATCM12_R8.txt",
]


# calculate the cosine similarity between each WP and Rec for each model
# ---

rec_documents = [
    Path(os.path.join(dname_rec, fname_rec)).read_text()
    for fname_rec in fname_recs
]

wp_documents = [
    Path(os.path.join(dname_wp, fname_wp)).read_text()
    for fname_wp in fname_wps
]


vo = voyageai.Client() # automatically use environment var VOYAGE_API_KEY
model_name = "voyage-2"

for model_name in model_names:

    wp_embeddings = vo.embed(
        wp_documents, model=model_name, input_type="document"
    ).embeddings

    rec_embeddings = vo.embed(
        rec_documents, model=model_name, input_type="document"
    ).embeddings

    # cosine similarity between them
    # TODO dot product is also possible here
    similaritys = cosine_similarity(wp_embeddings, rec_embeddings)

    # save to csv
    df = pd.DataFrame(
        similaritys, 
        columns = ["Rec_" + str(i) for i in range(1, nbr_recs+1)],
        index = ["WP_" + str(i) for i in range(1, nbr_wps+1)]
    )
    df.to_csv(fname_out_fnc(model_name), index=True)

"""


    print(f"Working on {model_name}")
    model = SentenceTransformer(model_name)

    # get embeddings of each Recommendation and Working Paper

    print("Getting Recommendation embeddings...")
    rec_embeddings = [
        model.encode(
            Path(os.path.join(dname_rec, fname_rec)).read_text()
        )
        for fname_rec in fname_recs
    ]

    print("Getting Working Paper embeddings...")
    wp_embeddings = [
        model.encode(
            Path(os.path.join(dname_wp, fname_wp)).read_text()
        )
        for fname_wp in fname_wps
    ]

    # cosine similarity between them
    similaritys = cosine_similarity(wp_embeddings, rec_embeddings)


    # save to csv
    df = pd.DataFrame(
        similaritys, 
        columns = ["Rec_" + str(i) for i in range(1, nbr_recs+1)],
        index = ["WP_" + str(i) for i in range(1, nbr_wps+1)]
    )
    df.to_csv(fname_out_fnc(model_name), index=True)
"""