# plot how the cosine similarities between WPs and Recs compare to
# the reality of which WP contributed to which Rec

import pandas as pd
import matplotlib.pyplot as plt

# parameters
# ---

# number of working papers and recommendations
nbr_wps = 20
nbr_recs = 8

fname_in_true = "../../results/compare_vector_embeddings_1983/similarities_by_hand.csv"
fname_in_fnc = lambda source_name, model_name: f"../../results/compare_vector_embeddings_1983/similarities_{source_name}_{model_name}.csv" # noqa
fname_out_fnc = lambda rec_nbr, source_name, model_name: f"../../results/compare_vector_embeddings_1983/similarities_rec_{rec_nbr}_{source_name}_{model_name}.pdf" # noqa

# structure is {source_1_name: [model_1_name, model_2_name, ...], source_2_name: [...], ...}
model_sources_names = {
    # "voyage": ["voyage-2", "voyage-law-2"],
    # "hugface": ["multi-qa-mpnet-base-dot-v1", "all-MiniLM-L6-v2"],
    "parsa": ["unknown"],
}

# read in the true contributions I found by hand
df_true = pd.read_csv(fname_in_true, index_col=0)

# for each model
# ---

for source_name, model_names in model_sources_names.items():
    for model_name in model_names:

        # import the relevant data
        df = pd.read_csv(fname_in_fnc(source_name, model_name), index_col=0)

        for rec_nbr in range(1, nbr_recs+1):

            rec_header = f"Rec_{rec_nbr}"

            # which wps contributed
            # contributeds = [True if v == 1 else False for v in df_true[rec_header]]
            contributeds = [True if df_true.loc[wp][rec_header] == 1 else False for wp in df.index]

            # get and sort the working papers high to low
            wps_sims_conts = sorted(list(zip(df.index, df[rec_header], contributeds)), key = lambda v: v[1], reverse=True)
            wps, sims, conts = zip(*wps_sims_conts)
            wps = [s[3:] for s in wps]

            # plot it
            bar_colours = ["red" if contributed else "blue" for contributed in conts]
            plt.figure(figsize=(4.5, 2))
            plt.bar(wps, sims, color = bar_colours, alpha=0.7)
            # plt.ylabel(f"cosine similarity to Rec. {rec_nbr}")
            # plt.xlabel('Working Paper')
            plt.ylim((0, 1))
            plt.xlim((-0.5, len(wps) - 0.5))
            plt.yticks([0, 1], ["0", "1"])
            plt.tight_layout()
            plt.savefig(fname_out_fnc(rec_nbr, source_name, model_name))
            plt.close("all")