# BMI = weight / ( height ** 2 )
user_weight = float(input("Enter your weight in kg: "))
user_height = float(input("Enter your height in m: "))
user_BMI = user_weight / (user_height) ** 2
# Convert numbers to strings before printing
print("Your BMI:" + str(user_BMI))

# Slim:user_BMI <= 18.5
# Normal: 18.5 < user_BMI <= 25
# Slightly overweight: 25 < user_BMI <= 30
# Obesity: user_BMI > 30
if user_BMI < 18.5: