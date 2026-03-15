import pyarrow as pa
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile(r"fineweb\downloaded\data\CC-MAIN-2013-48\000_00025.parquet")

print(parquet_file.schema.names)
print(parquet_file.schema)
print(parquet_file.metadata.num_rows)

for batch in parquet_file.iter_batches(batch_size=100):  # 每次处理1000行
    df_batch = batch.to_pandas()
    print(df_batch.head(100))
    pq.write_table(pa.Table.from_batches([batch]), f"preview.parquet", compression='zstd', compression_level=3)
    exit()