import pandas as pd
import numpy as np
from genotype_caller import call_genotype, ALLELES
from somatic_variant_caller import somatic_vaf
from per_base_error_estimate import error_estimate

GENOTYPE_KEY = "genotype"
ERROR_ESTIMATE_KEY = "error_estimate"
POS_KEY = "pos"
BASE_COUNT_COLUMNS = [POS_KEY] + ALLELES
VAF_HEADER = [f"VAF_{allele}" for allele in ALLELES]


class Dataset(object):
    def __init__(self, sample_name_to_filepaths, controls=None, sep='\t'):
        self.sample_name_to_filepaths = sample_name_to_filepaths
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
        self.df_error = None
        self.df_somatic = None
        self.df_genotype = None
        self.df_base_count = None
        self._common_positions = None
        self.per_base_error = None

    def __getitem__(self, sample_name):
        return self.dataset[sample_name]

    def get_df_base_count(self):
        for sample in self.dataset.values():
            sample.get_df_base_count()
        self.df_base_count = pd.concat([self.dataset[sample_name].df_base_count for sample_name in self.sample_names],
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


class Sample(object):
    def __init__(self, sample_name, filepath, is_control=False, sep='\t'):
        self.sample_name = sample_name
        self.filepath = filepath
        self.is_control = is_control
        self.sep = sep
        self.df_base_count = None
        self.df_genotype = None
        self.df_somatic = None
        self.df = None

    def get_df_base_count(self, reread=False):
        if self.df_base_count is not None and not reread:
            return self.df_base_count
        df = pd.read_csv(self.filepath, sep=self.sep)
        self.df_base_count = df
        assert list(df.columns) == BASE_COUNT_COLUMNS, (f"Non-uniform base_count_header. "
                                                        f"Expected: {BASE_COUNT_COLUMNS}\n"
                                                        f"Given: {df.columns}.")
        return df

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
