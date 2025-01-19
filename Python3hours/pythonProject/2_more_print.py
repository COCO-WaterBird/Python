import math
from operator import length_hint

print("hello " + "code " + "lol")

print('Let\'s go')
print("\n")
print("first column\nsecond column")

print(" He said 'good!' ")

print("""1column
2column
3column
""")

greet = "Number"
greet_number = greet
greet_Name = "Big"
greet = greet_Name
print(greet + "Third")
print(greet + "Four")

a = -1
b = -2
c = 3
#** square
print((-b + (b ** 2 - 4 * a * c)**(1/2)) / (2 * a))
print((-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a))


# length

s = "Hello World!"
print(len(s))
print(s[0])
print(s[len(s) - 1])

# bool
b1 = True
b2 = False

# None
n = None

# type
print(type(s))
print(type(b1))
print(type(n))
print(type(1.5))

# BMI = weight / ( height ** 2 )
user_weight = float(input("Enter your weight in kg: "))
user_height = float(input("Enter your height in m: "))
user_BMI = user_weight / (user_height) ** 2
# Convert numbers to strings before printing
print("Your BMI:" + str(user_BMI))


