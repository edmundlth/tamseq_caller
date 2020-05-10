import pandas as pd
import numpy as np
from genotype_caller import call_genotype, ALLELES
from somatic_variant_caller import somatic_vaf
from per_base_error_estimate import error_estimate
from dirichlet_fitting import pseudocount_estimate
from scipy.special import gamma, loggamma, digamma, polygamma
from scipy import stats

GENOTYPE_KEY = "genotype"
ERROR_ESTIMATE_KEY = "error_estimate"
POS_KEY = "pos"
RAW_COLUMNS = ['chrom', 'pos', 'sample', 'A', 'T', 'G', 'C', 'unfiltered coverage',
               'filtered coverage', 'failed nm count', 'failed amplicon location',
               'failed mapping quality', 'failed align length', 'failed base quality',
               'failed valid DNA base', 'base absent', 'base not overlapped',
               'failed overlapping positions']

BASE_COUNT_COLUMNS = [POS_KEY] + ALLELES
VAF_HEADER = [f"VAF_{allele}" for allele in ALLELES]


class Dataset(object):
    def __init__(self,
                 sample_name_to_filepaths,
                 base_counter_columns=RAW_COLUMNS,
                 controls=None,
                 sep='\t'):
        self.sample_name_to_filepaths = sample_name_to_filepaths
        self.base_counter_columns = base_counter_columns
        self.sample_names = sorted(self.sample_name_to_filepaths.keys())
        self.controls = controls
        assert set(self.controls).issubset(set(self.sample_names)), "Control set is not subset of sample set."
        self.sep = sep
        self.dataset = {
            sample_name: Sample(sample_name,
                                filepath,
                                is_control=(sample_name in self.controls),
                                sep=self.sep)
            for sample_name, filepath in sample_name_to_filepaths.items()
        }
        self.df_raw = None
        self.df_error = None
        self.df_somatic = None
        self.df_genotype = None
        self.df_base_count = None
        self._common_positions = None
        self.per_base_error = None
        self.df_pseudocount = None
        self.df_case = None

    def __getitem__(self, sample_name):
        return self.dataset[sample_name]

    def get_df_base_count(self):
        for sample in self.dataset.values():
            sample.get_df_base_count()
        self.df_base_count = pd.concat([self.dataset[sample_name].df_base_count for sample_name in self.sample_names],
                                       keys=self.sample_names, names=["sample", "idx"])
        return self

    def get_df_raw(self):
        for sample in self.dataset.values():
            sample.get_df_raw()
        self.df_raw = pd.concat([self.dataset[sample_name].df_raw for sample_name in self.sample_names],
                                keys=self.sample_names)
        return self

    def get_dataset_common_positions(self, recalc=False):
        if self._common_positions is not None and not recalc:
            return self._common_positions
        positions = self.get_common_positions()
        self._common_positions = positions
        return positions

    def get_common_positions(self, sample_names=None):
        assert self.df_base_count is not None, "Data have not been read from file."
        if type(sample_names) == str and sample_names.lower() in ["control", "controls"]:
            sample_names = self.controls
        elif sample_names is None:
            sample_names = self.sample_names

        positions = set()
        for sample_name in sample_names:
            df = self.df_base_count.loc[sample_name]
            nonzero_coverage = np.sum(df[ALLELES], axis=1) > 0
            pos_set = set(df[nonzero_coverage][POS_KEY])
            positions = pos_set.intersection(positions) if positions else pos_set
            assert len(positions) > 0, "Empty position intersection."
        return positions

    def compute_error_estimates(self):
        error_rec = {}
        positions = self.get_common_positions(sample_names=self.controls)

        for sample_name in self.controls:
            error_rec[sample_name] = {}
            df = self.dataset[sample_name].df_base_count.set_index(POS_KEY).loc[positions].reset_index()
            for row in df[BASE_COUNT_COLUMNS].itertuples():
                pos = row[1]
                n_obs = np.array(row[2:])
                if not np.any(n_obs):  # if all zeros, i.e. coverage is zero at this position
                    continue
                er = error_estimate(n_obs)
                error_rec[sample_name][pos] = er
        self.df_error = pd.DataFrame.from_dict(error_rec, orient="columns")
        self.per_base_error = self.df_error.mean(axis=1)
        return self.df_error

    def call_genotype(self):
        for sample in self.dataset.values():
            sample.call_genotype()

        self.df_genotype = pd.concat([self.dataset[sample_name].df_genotype for sample_name in self.sample_names],
                                     keys=self.sample_names)
        return self

    def get_somatic_vaf(self):
        for sample in self.dataset.values():
            sample.get_somatic_vaf(position_error_esitmate=self.per_base_error)
        df = pd.concat([self.dataset[sample_name].df_somatic for sample_name in self.sample_names],
                       keys=self.sample_names)
        df["AD"] = df[ALLELES].sum(axis=1)
        df["max_vaf"] = df[VAF_HEADER].max(axis=1)
        df["MAF"] = np.sort(df[ALLELES], axis=1)[:, -2] / df["AD"]
        self.df_somatic = df
        return self.df_somatic

    def fit_control_dirichlet_model(self,
                                    eps=1e-6,
                                    damping=1e-2,
                                    max_num_iter=10000):
        df_data = self.df_base_count.reset_index().set_index("pos")
        df = df_data.reset_index().set_index("sample").loc[
            self.controls,
            ['pos'] + list(ALLELES)
        ].set_index('pos').sort_index().copy(deep=True)
        result = pseudocount_estimate(df, eps=eps, damping=damping, max_num_iter=max_num_iter)
        self.df_pseudocount = pd.DataFrame.from_dict(result, orient="index")
        self.df_pseudocount.columns = ["iter", "res"] + [f"alpha_{a}" for a in ALLELES]
        return result

    def case_samples_variant_calling(self, df_ref_allele, conf_level=0.99):
        case_samples = list(set(self.sample_name_to_filepaths.keys()) - set(self.controls))
        df_data = self.df_base_count.reset_index().set_index("pos").join(df_ref_allele, how='left', on="pos")

        df_case = (df_data[df_data[ALLELES].sum(axis=1) > 0].reset_index().set_index('sample')
                   .loc[case_samples]
                   .reset_index()
                   .set_index("pos")
                   .join(self.df_pseudocount, how="left"))
        self.df_case = df_case
        alpha_alleles = [f"alpha_{a}" for a in ALLELES]
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


        df_case.index.name = "pos"
        #df_case.set_index("pos", inplace=True)

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

        a_param = np.array(df_case[ALLELES])[
                      np.arange(df_case.shape[0]),
                      np.argmax(np.array(ALLELES) == np.array(df_case["alt"]).reshape(-1, 1), axis=1)] + 1 / 2
        a_param[df_case["alt"] == '.'] *= 0
        b_param = df_case["n"] + 2 - a_param
        interval = stats.beta.interval(conf_level, a_param, b_param)

        df_case["alt_vaf_low"] = interval[0]
        df_case["alt_vaf_high"] = interval[1]
        df_case["alt_vaf_lower_bound"] = stats.beta.ppf(1 - conf_level, a_param, b_param)

        noise_level = np.clip(df_case[alpha_alleles] - 1, 0, np.inf) / np.array(
            df_case[alpha_alleles].sum(axis=1) - 4).reshape(-1, 1)
        noise_level = np.array(noise_level)[np.arange(noise_level.shape[0]),
                                            np.argmax(np.array(ALLELES) == np.array(df_case["alt"]).reshape(-1, 1),
                                                      axis=1)]
        noise_level[df_case["alt"] == '.'] *= 0
        df_case["alt_vaf_noise_corrected"] = df_case["alt_vaf"] - noise_level
        self.df_case = df_case
        return df_case


########################################################################################
# Sample Object
########################################################################################

class Sample(object):
    def __init__(self, sample_name, filepath, is_control=False, sep='\t'):
        self.sample_name = sample_name
        self.filepath = filepath
        self.is_control = is_control
        self.sep = sep
        self.df_raw = None
        self.df_base_count = None
        self.df_genotype = None
        self.df_somatic = None
        self.df = None

    def get_df_raw(self, reread=False):
        if self.df_raw is not None and not reread:
            return self.df_raw
        df = pd.read_csv(self.filepath, sep=self.sep)
        # assert list(df.columns) == RAW_COLUMNS, (f"Non-uniform base_count_header. "
        #                                          f"Expected: {RAW_COLUMNS}\n"
        #                                          f"Given: {df.columns}.")
        self.df_raw = df
        return df

    def get_df_base_count(self, recalc=False):
        if self.df_base_count is not None and not recalc:
            return self.df_base_count
        self.df_base_count = self.get_df_raw()[BASE_COUNT_COLUMNS]
        return self.df_base_count

    def get_df_genotype(self, recalc=False):
        if self.df_genotype is not None and not recalc:
            return self.df_genotype
        return self.call_genotype()

    def call_genotype(self, df_base_count=None):
        if not df_base_count:
            df = self.get_df_base_count()
        else:
            df = df_base_count

        call_data = {}
        for row in df[BASE_COUNT_COLUMNS].itertuples():
            pos = row[1]
            gt_data = call_genotype(row[2:])
            call_data[pos] = gt_data
        df_call = pd.DataFrame.from_dict(call_data, orient='index', columns=["genotype", "DKLmin", "DKL2"])
        self.df_genotype = df[BASE_COUNT_COLUMNS].set_index(POS_KEY).join(df_call)
        return self.df_genotype

    def get_somatic_vaf(self, position_error_esitmate=0.001, recalc=False, control_calc=False):
        if self.is_control and not control_calc:
            return None

        if self.df_somatic is not None and not recalc:
            return self.df_somatic

        df = self.get_df_genotype()
        if type(position_error_esitmate) == float:
            df[ERROR_ESTIMATE_KEY] = np.ones(df.shape[0]) * position_error_esitmate
        elif type(position_error_esitmate) == pd.Series:
            position_error_esitmate.name = ERROR_ESTIMATE_KEY
            positions = np.array(list(set(df.index).intersection(set(position_error_esitmate.keys()))))
            positions.sort()
            df = df.loc[positions].join(position_error_esitmate, how="inner", on=POS_KEY).dropna()
        else:
            df[ERROR_ESTIMATE_KEY] = position_error_esitmate

        vafs = []
        for row in df.itertuples():
            vafs.append(somatic_vaf(np.array(row[1:5]), row[8], row[5]))
        df_vaf = pd.DataFrame(vafs, columns=VAF_HEADER, index=df.index)
        self.df_somatic = df.join(df_vaf, how="inner")
        return self.df_somatic
