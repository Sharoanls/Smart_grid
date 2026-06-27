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

# Add dummy features for training AI models (Voltage and Current measurements and angles)
np.random.seed(42)

# Helper function to assign magnitudes and angles
def assign_features(mask, v_mags, v_angs, i_mags, i_angs):
    n = mask.sum()
    dataset.loc[mask, "V_A"] = np.random.normal(v_mags[0], 0.05, n)
    dataset.loc[mask, "V_B"] = np.random.normal(v_mags[1], 0.05, n)
    dataset.loc[mask, "V_C"] = np.random.normal(v_mags[2], 0.05, n)
    
    dataset.loc[mask, "V_A_angle"] = np.random.normal(v_angs[0], 2.0, n)
    dataset.loc[mask, "V_B_angle"] = np.random.normal(v_angs[1], 2.0, n)
    dataset.loc[mask, "V_C_angle"] = np.random.normal(v_angs[2], 2.0, n)

    dataset.loc[mask, "I_A"] = np.random.normal(i_mags[0], 0.1 * (i_mags[0] if i_mags[0] > 1 else 1), n)
    dataset.loc[mask, "I_B"] = np.random.normal(i_mags[1], 0.1 * (i_mags[1] if i_mags[1] > 1 else 1), n)
    dataset.loc[mask, "I_C"] = np.random.normal(i_mags[2], 0.1 * (i_mags[2] if i_mags[2] > 1 else 1), n)

    dataset.loc[mask, "I_A_angle"] = np.random.normal(i_angs[0], 5.0, n)
    dataset.loc[mask, "I_B_angle"] = np.random.normal(i_angs[1], 5.0, n)
    dataset.loc[mask, "I_C_angle"] = np.random.normal(i_angs[2], 5.0, n)

# Normal conditions: Voltages ~1.0 pu, angles 0, -120, 120. Currents lag by 15 deg.
normal_mask = dataset["Fault_Type"] == "Normal"
assign_features(
    normal_mask, 
    v_mags=[1.0, 1.0, 1.0], v_angs=[0, -120, 120], 
    i_mags=[0.5, 0.5, 0.5], i_angs=[-15, -135, 105]
)

# L-G Fault (Phase A to Ground): V_A drops, I_A spikes and becomes highly inductive (lags by 75 deg)
lg_mask = dataset["Fault_Type"] == "L-G"
assign_features(
    lg_mask, 
    v_mags=[0.1, 1.15, 1.15], v_angs=[0, -120, 120], 
    i_mags=[5.0, 0.5, 0.5], i_angs=[-75, -135, 105]
)

# LL-G Fault (Phase A and B to Ground): V_A, V_B drop. I_A, I_B spike and lag.
llg_mask = dataset["Fault_Type"] == "LL-G"
assign_features(
    llg_mask, 
    v_mags=[0.15, 0.15, 1.2], v_angs=[-15, -105, 120], 
    i_mags=[6.0, 6.0, 0.5], i_angs=[-90, -180, 105]
)

# LLL Fault (Three-phase without ground): Symmetrical drop, high currents
lll_mask = dataset["Fault_Type"] == "LLL"
assign_features(
    lll_mask, 
    v_mags=[0.4, 0.4, 0.4], v_angs=[0, -120, 120], 
    i_mags=[7.0, 7.0, 7.0], i_angs=[-80, -200, 40]
)

# LLL-G Fault (Three-phase to ground): Severe symmetrical drop
lllg_mask = dataset["Fault_Type"] == "LLL-G"
assign_features(
    lllg_mask, 
    v_mags=[0.0, 0.0, 0.0], v_angs=[0, -120, 120], 
    i_mags=[10.0, 10.0, 10.0], i_angs=[-85, -205, 35]
)  

# Save to CSV
dataset.to_csv("microgrid_fault_dataset.csv", index=False)
print("Dataset generated successfully: microgrid_fault_dataset.csv")
