import pandas as pd
import json
# load xlsx as pandas dataframe
df = pd.read_excel("Results_.xlsx")

# remove unnecessary columns (TODO LATER)


# import JSON file
with open("results/sleep_results.json", "r") as file:
    data = json.load(file)

# iterate through the data and add it to the workbook
for item in data:
    dataset = item["dataset"].lower()
    sleep_pct = item["sleep_interval_pct"]
    seed = item["seed"]
    acc = item["final_test_acc"]
    model = "snntorch"

    # convert sleep_pct to whole percentage numbers
    if sleep_pct != 0.0:
        sleep_pct = int(sleep_pct * 100)
    else:
        sleep_pct = int(0)

    # assign data based on labels
    mask = (
        (df['Sleep_duration'] == sleep_pct)
        & (df['Seed'] == seed)
        & (df['Model'] == model)
        & (df['Run'] == 1.0)
    )
    print("Seed: ", seed, "Dataset: ", dataset, "Sleep: ", sleep_pct, "Model: ", model, "Run: ", 1.0)
    print("Run: ", any(df["Run"] == 1.0))
    print("Seed and Run: ", any((df["Seed"] == seed) & (df["Run"] == 1.0)))
    print("Model and Run: ", any((df["Model"] == model) & (df["Run"] == 1.0)))
    print("Sleep and Model and Run: ", any((df["Sleep_duration"] == sleep_pct) & (df["Model"] == model) & (df["Run"] == 1.0)))
    print("Sleep and Seed and Model and Run: ", any((df["Sleep_duration"] == sleep_pct) & (df["Seed"] == seed) & (df["Model"] == model) & (df["Run"] == 1.0)))
    print("Mask: ", any(mask))
    
    df.loc[mask, dataset] = acc

# save the dataframe
df.to_excel("Results_.xlsx", index=False)

