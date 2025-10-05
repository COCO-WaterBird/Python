def load(data_frame, file_name):
    # Write cleaned_data to a CSV using file_name
    data_frame.to_csv(file_name, index=False)
    print(f"Successfully loaded data to {file_name}")

# Extract data from raw_data.csv
extracted_data = extract(file_name="raw_data.csv")

# Transform extracted_data using transform() function
transformed_data = transform(data_frame=extracted_data)

# Load transformed_data to transformed_data.csv
load(data_frame=transformed_data, file_name="transformed_data.csv")


# Complete building the transform() function
def transform(source_table, target_table):
  data_warehouse.execute(f"""
  CREATE TABLE {target_table} AS
      SELECT
          CONCAT("Product ID: ", product_id),
          quantity * price
      FROM {source_table};
  """)

extracted_data = extract(file_name="raw_sales_data.csv")
load(data_frame=extracted_data, table_name="raw_sales_data")

# Populate total_sales by transforming raw_sales_data
transform(source_table="raw_sales_data", target_table="total_sales")


def extract(file_name):
    return pd.read_csv(file_name)


def transform(data_frame):
    return data_frame.loc[:, ["industry_name", "number_of_firms"]]


def load(data_frame, file_name):
    data_frame.to_csv(file_name)


extracted_data = extract(file_name="raw_industry_data.csv")
transformed_data = transform(data_frame=extracted_data)

# Pass the transformed_data DataFrame to the load() function
load(data_frame=transformed_data, file_name="number_of_firms.csv")
