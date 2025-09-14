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


