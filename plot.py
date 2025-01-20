import matplotlib.pyplot as plt

GSM8K = [41.0, 55.7, 58.7, 63.7]
MATH = [12.3, 22.3, 24.0, 37.7]

compute = [1, 4, 9, 49]
plt.figure(figsize=(12,8))

plt.plot(compute, GSM8K, marker="o", label="GSMK8K", linewidth=2)

plt.xlabel('compute (B²)')
plt.ylabel("accuracy")
plt.grid(True, linestyle="--", alpha=0.7)

plt.savefig("GSM8K_scaling.pdf")

