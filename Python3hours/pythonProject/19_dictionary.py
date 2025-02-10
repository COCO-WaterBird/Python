slang_dict = {"MDY":"Month Day Year",
              "TBD":"To Be Determine"}





query = input("Please enter a word: ")
if query in slang_dict:
    print(f"{query} is in the dictionary")
    print(slang_dict[query])
else:
    print(f"{query} is not in the dictionary")
    print("Current dict number is " + str(len(slang_dict)))
