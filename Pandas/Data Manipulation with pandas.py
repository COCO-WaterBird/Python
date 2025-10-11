# Sort homelessness by descending family members
homelessness_fam = homelessness.sort_values('family_members', ascending =[False])

print(homelessness_fam.head())

# Sort homelessness by region, then descending family members
homelessness_reg_fam = homelessness.sort_values(['region','family_members'],ascending=[True,False])

# Print the top few rows
print(homelessness_reg_fam.head())

# Create a Series containing only the individuals column from homelessness
individuals = homelessness['individuals']

# Print the first few rows
print(individuals.head())

# Create a DataFrame with just the individuals and state columns from homelessness
ind_state = homelessness[['individuals','state']]

# Print the first few rows
print(ind_state.head())

# Filter homelessness for rows where individuals is greater than 10000
ind_gt_10k = homelessness[homelessness['individuals']>10000]

# Print the result
print(ind_gt_10k)

# Filter homelessness for rows where region is Mountain, assign result to mountain_reg
mountain_reg = homelessness[homelessness['region'] == 'Mountain']

# Print the filtered DataFrame
print(mountain_reg)

# Filter homelessness for rows with family_members < 1000 and region is "Pacific". Assign to fam_lt_1k_pac
fam_lt_1k_pac = homelessness[(homelessness['family_members']<1000) & (homelessness['region'] == 'Pacific')]

# Print the result
print(fam_lt_1k_pac)

# Mojave Desert states list
canu = ["California", "Arizona", "Nevada", "Utah"]

# Filter homelessness for states in canu, assign to mojave_homelessness
mojave_homelessness =homelessness[homelessness['state'].isin(canu)]

# Print the result
print(mojave_homelessness)

# Add a column to homelessness with the sum of individuals and family_members
homelessness['total'] = homelessness['individuals'] + homelessness['family_members']

# Add a column with the proportion of total homeless to the state population
homelessness['p_homeless'] = homelessness['total'] / homelessness['state_pop']

# Print the updated DataFrame
print(homelessness)

# Add a column for homeless individuals per 10k state population
homelessness['indiv_per_10k'] = homelessness['individuals']/homelessness['state_pop'] * 10000

# Subset rows with indiv_per_10k greater than 20
high_homelessness = homelessness[homelessness['indiv_per_10k'] > 20]

# Sort the subset by indiv_per_10k in descending order
high_homelessness_srt = high_homelessness.sort_values(['indiv_per_10k'],ascending=[False])
# Select only the state and indiv_per_10k columns
result = high_homelessness_srt[['state','indiv_per_10k']]

# Print the result
print(result)


# Print the first few rows of sales
print(sales.head())

# Print column info for sales
print(sales.info())

# Print the mean of weekly_sales
print(sales['weekly_sales'].mean())

# Print the median of weekly_sales
print(sales['weekly_sales'].median())

# Print the maximum of the date column
print(sales['date'].max())

# Print the minimum of the date column
print(sales['date'].min())


# A custom IQR function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Print the IQR of temperature_c from sales
print(sales['temperature_c'].agg(iqr))

# A custom IQR function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Print the IQR of temperature_c, fuel_price_usd_per_l, & unemployment from sales
print(sales[['temperature_c','fuel_price_usd_per_l','unemployment']].agg(iqr))

# Define a custom function to calculate IQR
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Print the IQR and median for the columns temperature_c, fuel_price_usd_per_l, and unemployment in sales
print(sales[['temperature_c','fuel_price_usd_per_l','unemployment']].agg([iqr,'median']))

# Sort sales_1_1 by date
sales_1_1 = sales_1_1.sort_values('date')

# Add cumulative sum of weekly_sales as cum_weekly_sales column
sales_1_1['cum_weekly_sales'] = sales_1_1['weekly_sales'].cumsum()

# Add cumulative max of weekly_sales as cum_max_sales column
sales_1_1['cum_max_sales'] = sales_1_1['weekly_sales'].cummax()

# Print the date, weekly_sales, cum_weekly_sales, and cum_max_sales columns
print(sales_1_1[['date','weekly_sales', 'cum_weekly_sales','cum_max_sales']])

# Drop duplicate store/type combinations
store_types = sales.drop_duplicates(subset=['store','type'])
print(store_types.head())

# Drop duplicate store/department combinations
store_depts = sales.drop_duplicates(subset=['store','department'])
print(store_depts.head())

# Subset sales for rows where is_holiday is True, then drop duplicate dates
holiday_dates = sales[sales['is_holiday']==True].drop_duplicates(subset = ['date'])

# Print the date column of holiday_dates
print(holiday_dates['date'])


# Count the number of stores of each type
store_counts = store_types['type'].value_counts()
print(store_counts)

# Get the proportion of stores of each type
store_props = store_types['type'].value_counts(normalize=True)
print(store_props)

# Count the number of stores for each department and sort
dept_counts_sorted = store_depts['department'].value_counts().sort_values(ascending=False)
print(dept_counts_sorted)

# Get the proportion of stores in each department and sort
dept_props_sorted = store_depts['department'].value_counts(normalize=True).sort_values(ascending=False)
print(dept_props_sorted)

# Calculate total weekly sales
sales_all = sales['weekly_sales'].sum()

# Calculate total weekly sales for type A stores
sales_A = sales[sales['type'] == 'A']['weekly_sales'].sum()

# Calculate total weekly sales for type B stores
sales_B = sales[sales['type'] == 'B']['weekly_sales'].sum()

# Calculate total weekly sales for type C stores
sales_C = sales[sales['type'] == 'C']['weekly_sales'].sum()

# Create a list of totals for A, B, and C, divide by overall total. Print the result.
sales_propn_by_type = [sales_A/sales_all,sales_B/sales_all,sales_C/sales_all]
print(sales_propn_by_type)


# From previous step
sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# Group sales by type and is_holiday, take sum of weekly_sales
sales_by_type_is_holiday = sales.groupby(['type','is_holiday'])['weekly_sales'].sum()
print(sales_by_type_is_holiday)

# Aggregate weekly_sales for each store type: min, max, mean, and median
sales_stats = sales.groupby('type')['weekly_sales'].agg(['min','max','mean','median'])

# Print the sales_stats DataFrame
print(sales_stats)

# Aggregate unemployment and fuel_price_usd_per_l for each store type: min, max, mean, and median
unemp_fuel_stats = sales.groupby('type')[['unemployment','fuel_price_usd_per_l']].agg(['min','max','mean','median'])

# Print the unemp_fuel_stats DataFrame
print(unemp_fuel_stats)

# Get the mean weekly_sales by type using a pivot table
mean_sales_by_type = sales.pivot_table(
    values='weekly_sales',
    index='type'

)

# Print mean_sales_by_type
print(mean_sales_by_type)

# Pivot for mean and median weekly_sales for each store type
mean_med_sales_by_type = sales.pivot_table(
    values = 'weekly_sales',
    index = 'type',
    aggfunc = ['mean','median']
)

# Print mean_med_sales_by_type
print(mean_med_sales_by_type)

# Pivot sales to get mean of weekly_sales by type and is_holiday
mean_sales_by_type_holiday = sales.pivot_table(
    values = 'weekly_sales',
    index = 'type',
    columns = 'is_holiday',
    aggfunc = 'mean'
)

# Print mean_sales_by_type_holiday
print(mean_sales_by_type_holiday)

# Print mean weekly_sales by department and type; fill missing values with 0
sales.pivot_table(
    values = 'weekly_sales',
    index = 'department',
    columns = 'type',
    aggfunc = 'mean',
    fill_value = 0
)

# Print the mean weekly_sales by department and type; fill missing values with 0s; sum all rows and cols
print(sales.pivot_table(
    values='weekly_sales',
    index='department',
    columns='type',
    aggfunc='mean',
    fill_value=0,
    margins=True
))

# Print temperatures
print(temperatures)

# Set the index of temperatures to city and assign to temperatures_ind
temperatures_ind = temperatures.set_index("city")

# Print temperatures_ind
print(temperatures_ind)

# Reset the index of temperatures_ind (keep index contents) and print
print(temperatures_ind.reset_index())

# Reset the index of temperatures_ind (drop index contents) and print
print(temperatures_ind.reset_index(drop=True))

# Create a list of cities to subset on
cities = ['London','Paris']

# Subset temperatures for cities using square brackets and print

print(temperatures[temperatures['city'].isin(cities)])

# Subset temperatures_ind for cities using .loc[] and print
print(temperatures_ind.loc[cities])
