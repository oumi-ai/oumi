import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("banking77-results.txt", sep=r"\s+", engine="python")

models = ["opus-4.6", "sonnet-4.6", "haiku-4.5"]
colors = ["tab:blue", "tab:orange", "tab:green"]

fig, ax = plt.subplots()

for model, color in zip(models, colors):
    subset = df[df["model"] == model].sort_values("examples")
    ax.plot(
        subset["examples"], subset["accuracy"], marker="o", label=model, color=color
    )

qwen_acc = df[df["model"] == "qwen3.5-0.8b-tuned"]["accuracy"].to_numpy()[0]
ax.axhline(y=qwen_acc, color="tab:red", linestyle="dotted", label="qwen3.5-0.8b-tuned")

ax.set_xlabel("Number of Examples")
ax.set_ylabel("Accuracy")
ax.set_xticks([0, 1, 3, 5])
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88))
plt.tight_layout()
plt.savefig("banking77-results.png", dpi=150)
plt.show()
