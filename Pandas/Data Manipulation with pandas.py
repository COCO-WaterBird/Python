# Sort homelessness by descending family members
homelessness_fam = homelessness.sort_values('family_members', ascending =[False])

print(homelessness_fam.head())

