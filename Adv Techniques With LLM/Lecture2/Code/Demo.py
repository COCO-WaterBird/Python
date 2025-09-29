import matplotlib.pyplot as plt

ans = [(0.3727259337902069, 0.8420291543006897), (0.368835985660553, 0.841996967792511), (0.3753706216812134, 0.8390715718269348), (0.37450721859931946, 0.8404539227485657)]
losses = [x[0] for x in ans]
accs = [x[1] for x in ans]
x = range(1, len(ans) + 1)
plt.figure(figsize=(10, 7))
plt.plot(x, losses, marker="s", label="Loss")
plt.plot(x, accs, marker="o", label="Accuracy")
plt.xlabel("Experiment Index")
plt.ylabel("Value")
plt.title("Loss & Accuracy in Weather")
plt.show()