
# Usage Example
```python
import os
import pandas as pd
from dataset import Dataset


datadir = "./data/tamseq_bams_pileup_count_stringent/"
extension = ".bam.tsv"
sample_name_to_filepaths = {filename.replace(extension, '') : os.path.join(datadir, filename)
                            for filename in os.listdir(datadir)}

controls = ["sample_name1", "sample_name2"]
dataset = Dataset(sample_name_to_filepaths, controls=controls)
dataset.get_df_base_count()
dataset.call_genotype()
df_error = dataset.compute_error_estimates()
df_genotype = dataset.df_genotype
df_somatic = dataset.get_somatic_vaf()
```