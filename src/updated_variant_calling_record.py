case_samples = list(set(sample_name_to_filepaths.keys()) - set(dataset.controls))
df_case = (df_data[df_data[ALLELES].sum(axis=1) > 0].reset_index().set_index('sample')
           .loc[case_samples]
           .reset_index()
           .set_index("pos")
           .join(df_pseudocount, how="left"))
vaf_cols = [f"VAF_{a}" for a in ALLELES]
alpha_modified_alleles = [f"alpha_modified_{a}" for a in ALLELES]
df_case = df_case[~np.any(np.isnan(df_case[alpha_alleles]), axis=1)]
for a in ALLELES:
    # Jeffrey's prior for multinomial
    df_case[f"alpha_modified_{a}"] = df_case[a] + 1 / 2  # df_case[f"alpha_{a}"]
df_case["alpha_modified_sum"] = df_case[alpha_modified_alleles].sum(axis=1)
effective_num_class = np.sum(df_case[ALLELES] != 0, axis=1)
for a in ALLELES:
    df_case[f"VAF_{a}"] = (np.clip(df_case[f"alpha_modified_{a}"] - 1, 0, np.inf)
                           / (df_case["alpha_modified_sum"] - effective_num_class))

df_case["n"] = df_case[ALLELES].sum(axis=1)
df_case["max_vaf_allele"] = np.array(ALLELES)[np.argmax(np.array(df_case[vaf_cols]), axis=1)]
idx = np.array(ALLELES) == np.array(df_case["ref"]).reshape(-1, 1)
df_case["ref_count"] = np.array(df_case[ALLELES])[np.newaxis, idx][0]
# dataset.per_base_error.name = "error_estimate"
# if "error_estimate" not in df_case.columns:
#    df_case = df_case.join(dataset.per_base_error)


header = list(df_case.columns)
allele_array = np.array(ALLELES)
REF_INDEX_KEY = header.index("ref") + 1
VAF_KEYS_START = header.index("VAF_A") + 1
VAF_KEYS_END = header.index("VAF_C") + 2
non_ref_mask = np.array([[0, 1, 1, 1],
                         [1, 0, 1, 1],
                         [1, 1, 0, 1],
                         [1, 1, 1, 0]]).astype(bool)
alt_alleles = []
alt_vafs = []
for row in df_case.itertuples():
    vaf = np.array(row[VAF_KEYS_START: VAF_KEYS_END])
    ref_index = np.argmax(np.array(ALLELES) == row[REF_INDEX_KEY])
    max_non_ref_vaf = np.max(vaf[non_ref_mask[ref_index]])
    if max_non_ref_vaf <= 1e-10 or np.any(np.isnan(vaf)):
        alt = '.'  # allele_array[ref_index]
        alt_vaf = 0
    else:
        alt = allele_array[vaf == max_non_ref_vaf][0]
        alt_vaf = max_non_ref_vaf
    alt_alleles.append(alt)
    alt_vafs.append(alt_vaf)
df_case["alt"] = alt_alleles
df_case["alt_vaf"] = alt_vafs

df_cosmic_pathogenic["alt"] = df_cosmic_pathogenic["Allele"]
df_gnomad["alt"] = df_gnomad["Alternate"]
df_gnomad["pos"] = df_gnomad.index

join_keys = ["pos", "alt"]
df_case.reset_index(inplace=True)
df_case.rename(mapper={"index": "pos"}, axis=1, inplace=True)
df_case = (df_case.join(df_cosmic_pathogenic.set_index(join_keys)[["cosmic count"]],
                        how='left',
                        on=join_keys)
           .join(df_gnomad.reset_index().set_index(join_keys)[["Allele Count", "Allele Frequency"]],
                 how='left',
                 on=join_keys,
                 rsuffix=" gnomAD"))
df_case.set_index("pos", inplace=True)

X = np.array(df_case[ALLELES])
alpha_0 = np.array(df_case[alpha_alleles])
a_0 = alpha_0.sum(axis=1)
a_1 = np.ones(df_case.shape[0]) * 2
t0 = np.zeros(df_case.shape[0])
t1 = loggamma(a_0)
t2 = loggamma(df_case["n"] + a_0)
t3 = loggamma(df_case["n"] + a_1)
t4 = np.sum(loggamma(X + 1 / 2) - loggamma(X + alpha_0) + loggamma(alpha_0) - loggamma(1 / 2), axis=1)
b = t0 - t1 + t2 - t3 + t4
b /= np.log(10)
df_case["bayes_factor"] = b

conf_level = 0.99
a_param = np.array(df_case[ALLELES])[
              np.arange(df_case.shape[0]),
              np.argmax(np.array(ALLELES) == np.array(df_case["alt"]).reshape(-1, 1), axis=1)] + 1 / 2
a_param[df_case["alt"] == '.'] *= 0
b_param = df_case["n"] + 2 - a_param
interval = stats.beta.interval(conf_level, a_param, b_param)

df_case["alt_vaf_low"] = interval[0]
df_case["alt_vaf_high"] = interval[1]
df_case["alt_vaf_lower_bound"] = stats.beta.ppf(1 - conf_level, a_param, b_param)

noise_level = np.clip(df_case[alpha_alleles] - 1, 0, np.inf) / np.array(df_case[alpha_alleles].sum(axis=1) - 4).reshape(
    -1, 1)
noise_level = np.array(noise_level)[np.arange(noise_level.shape[0]),
                                    np.argmax(np.array(ALLELES) == np.array(df_case["alt"]).reshape(-1, 1), axis=1)]
noise_level[df_case["alt"] == '.'] *= 0
df_case["alt_vaf_noise_corrected"] = df_case["alt_vaf"] - noise_level