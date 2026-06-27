import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

# Base counts
train_faults, val_faults, test_faults = 6999, 1500, 1501
train_normals, val_normals, test_normals = 1400, 300, 300

# Function to generate cases
def generate_cases(n, fault_type, windmill, resistance, bus_type, split):
    return pd.DataFrame({
        "Sample_ID": range(1, n+1), # Sample ID will be recalculated later to be unique
        "Fault_Type": [fault_type]*n,
        "Windmill": [windmill]*n,
        "Resistance": [resistance]*n,
        "Bus_Type": [bus_type]*n,
        "Split": [split]*n
    })

# Add normal cases
train_normal = generate_cases(train_normals, "Normal", "WM1", "N/A", "Generator_690V", "Train")
val_normal = generate_cases(val_normals, "Normal", "WM2", "N/A", "Collector_33kV", "Validation")
test_normal = generate_cases(test_normals, "Normal", "WM3", "N/A", "PCC_132kV", "Test")

# Add fault cases
fault_types = ["L-G", "LL-G", "LLL-G", "LLL"]
windmills = ["WM1", "WM2", "WM3"]
resistances = ["0.01", "10", "50"]
bus_types = ["Generator_690V", "Collector_33kV", "PCC_132kV"]

def generate_random_faults(n, split):
    df = pd.DataFrame({
        "Sample_ID": range(1, n+1),
        "Fault_Type": np.random.choice(fault_types, n),
        "Windmill": np.random.choice(windmills, n),
        "Resistance": np.random.choice(resistances, n),
        "Bus_Type": np.random.choice(bus_types, n),
        "Split": [split]*n
    })
    return df

train_fault = generate_random_faults(train_faults, "Train")
val_fault = generate_random_faults(val_faults, "Validation")
test_fault = generate_random_faults(test_faults, "Test")

# Combine into dataset
dataset = pd.concat([train_normal, val_normal, test_normal, train_fault, val_fault, test_fault], ignore_index=True)

# Randomize the dataset order
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

dataset["Sample_ID"] = range(1, len(dataset)+1)

# Add dummy features for training AI models (Voltage and Current measurements)
np.random.seed(42)
# Normal conditions have voltages near 1.0 pu and low currents
normal_mask = dataset["Fault_Type"] == "Normal"
dataset.loc[normal_mask, "V_A"] = np.random.normal(1.0, 0.05, normal_mask.sum())
dataset.loc[normal_mask, "V_B"] = np.random.normal(1.0, 0.05, normal_mask.sum())
dataset.loc[normal_mask, "V_C"] = np.random.normal(1.0, 0.05, normal_mask.sum())
dataset.loc[normal_mask, "I_A"] = np.random.normal(0.5, 0.1, normal_mask.sum())
dataset.loc[normal_mask, "I_B"] = np.random.normal(0.5, 0.1, normal_mask.sum())
dataset.loc[normal_mask, "I_C"] = np.random.normal(0.5, 0.1, normal_mask.sum())

# Fault conditions have dropped voltages and high currents, but depends on fault type
# L-G Fault (Phase A to Ground)
lg_mask = dataset["Fault_Type"] == "L-G"
dataset.loc[lg_mask, "V_A"] = np.random.normal(0.1, 0.05, lg_mask.sum()) # Dropped severely
dataset.loc[lg_mask, "V_B"] = np.random.normal(1.15, 0.05, lg_mask.sum()) # Swell on healthy phases
dataset.loc[lg_mask, "V_C"] = np.random.normal(1.15, 0.05, lg_mask.sum()) # Swell on healthy phases
dataset.loc[lg_mask, "I_A"] = np.random.normal(5.0, 0.5, lg_mask.sum())  # High fault current
dataset.loc[lg_mask, "I_B"] = np.random.normal(0.5, 0.1, lg_mask.sum())  # Normal load current
dataset.loc[lg_mask, "I_C"] = np.random.normal(0.5, 0.1, lg_mask.sum())  # Normal load current

# LL-G Fault (Phase A and B to Ground)
llg_mask = dataset["Fault_Type"] == "LL-G"
dataset.loc[llg_mask, "V_A"] = np.random.normal(0.15, 0.05, llg_mask.sum()) # Dropped severely
dataset.loc[llg_mask, "V_B"] = np.random.normal(0.15, 0.05, llg_mask.sum()) # Dropped severely
dataset.loc[llg_mask, "V_C"] = np.random.normal(1.2, 0.05, llg_mask.sum()) # Higher swell on healthy phase
dataset.loc[llg_mask, "I_A"] = np.random.normal(6.0, 0.5, llg_mask.sum())  # High fault current
dataset.loc[llg_mask, "I_B"] = np.random.normal(6.0, 0.5, llg_mask.sum())  # High fault current
dataset.loc[llg_mask, "I_C"] = np.random.normal(0.5, 0.1, llg_mask.sum())  # Normal load current

# LLL Fault (Three-phase without ground)
lll_mask = dataset["Fault_Type"] == "LLL"
dataset.loc[lll_mask, "V_A"] = np.random.normal(0.4, 0.05, lll_mask.sum()) # Symmetrical drop, but not zero
dataset.loc[lll_mask, "V_B"] = np.random.normal(0.4, 0.05, lll_mask.sum())
dataset.loc[lll_mask, "V_C"] = np.random.normal(0.4, 0.05, lll_mask.sum())
dataset.loc[lll_mask, "I_A"] = np.random.normal(7.0, 0.8, lll_mask.sum())  # Symmetrical high current
dataset.loc[lll_mask, "I_B"] = np.random.normal(7.0, 0.8, lll_mask.sum())  
dataset.loc[lll_mask, "I_C"] = np.random.normal(7.0, 0.8, lll_mask.sum())  

# LLL-G Fault (Three-phase to ground)
lllg_mask = dataset["Fault_Type"] == "LLL-G"
dataset.loc[lllg_mask, "V_A"] = np.random.normal(0.0, 0.02, lllg_mask.sum()) # Severe drop (near zero)
dataset.loc[lllg_mask, "V_B"] = np.random.normal(0.0, 0.02, lllg_mask.sum())
dataset.loc[lllg_mask, "V_C"] = np.random.normal(0.0, 0.02, lllg_mask.sum())
dataset.loc[lllg_mask, "I_A"] = np.random.normal(10.0, 1.2, lllg_mask.sum())  # Extreme current due to ground path
dataset.loc[lllg_mask, "I_B"] = np.random.normal(10.0, 1.2, lllg_mask.sum())  
dataset.loc[lllg_mask, "I_C"] = np.random.normal(10.0, 1.2, lllg_mask.sum())  

# Save to CSV
dataset.to_csv("microgrid_fault_dataset.csv", index=False)
print("Dataset generated successfully: microgrid_fault_dataset.csv")
