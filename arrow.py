import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Load the dataset from traffic-1.xlsx
df = pd.read_excel('traffic-1.xlsx')

# Calculate Total Vehicles
df['Total Vehicles'] = df['Car Count'] + df['Bike Count'] + df['Truck Count'] + df['Bus Count']

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Convert DataFrame to Arrow Table
table = pa.Table.from_pandas(df)

# Write Arrow Table to a Parquet file
pq.write_table(table, 'traffic-1.arrow')

print("\nDataFrame written to 'traffic-1.arrow' file in Arrow format.")

# Read Arrow Table from Parquet file
table_read = pq.read_table('traffic-1.arrow')

# Convert Arrow Table back to Pandas DataFrame
df_read = table_read.to_pandas()

print("\nDataFrame read from 'traffic-1.arrow' file:")
print(df_read)
