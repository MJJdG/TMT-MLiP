import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq

os.chdir("C:\\Users\\Ruben\Documents\\Master_AI_CNS\\MLiP")

for i in range(4):
    bestand = pd.read_parquet(f'bengaliai-cv19/train_image_data_{i}.parquet')
    
    num_parts = 6
    bins = np.array_split(bestand, num_parts)
    part = 0

    print(len(bins))

    for bin in bins:
        table = pa.Table.from_pandas(bin)
        pq.write_table(table, f'train_{i}_{part}.parquet')
        part += 1


