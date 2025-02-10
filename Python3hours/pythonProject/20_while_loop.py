print("Hi,I can get average")
total = 0
count = 0
user_input = input("Please enter anything, when you are done, enter end : ")
while user_input != "end":
    num = float(user_input)
    total = total + num
    total += num
    count = count + 1
    count += 1
    user_input = input("Please enter anything, when you are done, enter end : ")
if count == 0:
    result = 0
else:
    result = total/count
result = total / count
print("The average is " + str(result))