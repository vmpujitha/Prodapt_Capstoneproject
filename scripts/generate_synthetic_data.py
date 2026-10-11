import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def generate_synthetic_crop_dataset(output_path: str, num_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic Indian agricultural crop yield dataset
    matching the schema and statistical distributions of the Kaggle Crop Yield Data.
    """
    np.random.seed(seed)
    os.makedirs(Path(output_path).parent, exist_ok=True)

    crop_profiles = {
        "Rice": {
            "seasons": ["Kharif", "Whole Year"],
            "states": ["Punjab", "West Bengal", "Uttar Pradesh", "Andhra Pradesh", "Tamil Nadu", "Odisha", "Bihar"],
            "temp_mean": 27.0, "temp_std": 3.0,
            "rain_mean": 1350.0, "rain_std": 250.0,
            "humidity_mean": 82.0, "humidity_std": 6.0,
            "n_mean": 95.0, "n_std": 15.0,
            "p_mean": 45.0, "p_std": 10.0,
            "k_mean": 42.0, "k_std": 10.0,
            "ph_mean": 6.5, "ph_std": 0.5,
            "base_yield": 3.2, "yield_std": 0.7,
            "fertilizer_mean": 130.0, "pesticide_mean": 2.2
        },
        "Wheat": {
            "seasons": ["Rabi"],
            "states": ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Bihar"],
            "temp_mean": 18.5, "temp_std": 3.5,
            "rain_mean": 380.0, "rain_std": 90.0,
            "humidity_mean": 60.0, "humidity_std": 8.0,
            "n_mean": 110.0, "n_std": 20.0,
            "p_mean": 55.0, "p_std": 12.0,
            "k_mean": 45.0, "k_std": 10.0,
            "ph_mean": 6.8, "ph_std": 0.4,
            "base_yield": 3.8, "yield_std": 0.8,
            "fertilizer_mean": 140.0, "pesticide_mean": 1.8
        },
        "Maize": {
            "seasons": ["Kharif", "Rabi"],
            "states": ["Karnataka", "Madhya Pradesh", "Maharashtra", "Andhra Pradesh", "Rajasthan", "Bihar"],
            "temp_mean": 24.0, "temp_std": 3.0,
            "rain_mean": 650.0, "rain_std": 140.0,
            "humidity_mean": 68.0, "humidity_std": 8.0,
            "n_mean": 85.0, "n_std": 18.0,
            "p_mean": 48.0, "p_std": 10.0,
            "k_mean": 40.0, "k_std": 8.0,
            "ph_mean": 6.4, "ph_std": 0.5,
            "base_yield": 2.9, "yield_std": 0.6,
            "fertilizer_mean": 110.0, "pesticide_mean": 1.5
        },
        "Cotton": {
            "seasons": ["Kharif"],
            "states": ["Gujarat", "Maharashtra", "Telangana", "Andhra Pradesh", "Haryana", "Punjab"],
            "temp_mean": 28.5, "temp_std": 3.0,
            "rain_mean": 750.0, "rain_std": 160.0,
            "humidity_mean": 62.0, "humidity_std": 9.0,
            "n_mean": 105.0, "n_std": 22.0,
            "p_mean": 50.0, "p_std": 12.0,
            "k_mean": 48.0, "k_std": 12.0,
            "ph_mean": 7.3, "ph_std": 0.5,
            "base_yield": 1.9, "yield_std": 0.5,
            "fertilizer_mean": 145.0, "pesticide_mean": 3.4
        },
        "Sugarcane": {
            "seasons": ["Whole Year"],
            "states": ["Uttar Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat", "Andhra Pradesh"],
            "temp_mean": 30.0, "temp_std": 3.5,
            "rain_mean": 1600.0, "rain_std": 300.0,
            "humidity_mean": 76.0, "humidity_std": 7.0,
            "n_mean": 160.0, "n_std": 25.0,
            "p_mean": 70.0, "p_std": 15.0,
            "k_mean": 80.0, "k_std": 15.0,
            "ph_mean": 6.7, "ph_std": 0.5,
            "base_yield": 72.0, "yield_std": 12.0,
            "fertilizer_mean": 220.0, "pesticide_mean": 4.1
        },
        "Groundnut": {
            "seasons": ["Kharif", "Rabi"],
            "states": ["Gujarat", "Andhra Pradesh", "Tamil Nadu", "Rajasthan", "Karnataka", "Maharashtra"],
            "temp_mean": 26.5, "temp_std": 3.0,
            "rain_mean": 550.0, "rain_std": 120.0,
            "humidity_mean": 58.0, "humidity_std": 9.0,
            "n_mean": 35.0, "n_std": 10.0,
            "p_mean": 55.0, "p_std": 12.0,
            "k_mean": 50.0, "k_std": 10.0,
            "ph_mean": 6.2, "ph_std": 0.4,
            "base_yield": 1.6, "yield_std": 0.4,
            "fertilizer_mean": 75.0, "pesticide_mean": 1.3
        },
        "Chickpea": {
            "seasons": ["Rabi"],
            "states": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Uttar Pradesh", "Karnataka", "Andhra Pradesh"],
            "temp_mean": 20.0, "temp_std": 3.0,
            "rain_mean": 320.0, "rain_std": 70.0,
            "humidity_mean": 52.0, "humidity_std": 8.0,
            "n_mean": 30.0, "n_std": 8.0,
            "p_mean": 60.0, "p_std": 10.0,
            "k_mean": 38.0, "k_std": 8.0,
            "ph_mean": 7.1, "ph_std": 0.4,
            "base_yield": 1.1, "yield_std": 0.3,
            "fertilizer_mean": 65.0, "pesticide_mean": 1.1
        },
        "Mustard": {
            "seasons": ["Rabi"],
            "states": ["Rajasthan", "Haryana", "Madhya Pradesh", "Uttar Pradesh", "West Bengal", "Gujarat"],
            "temp_mean": 17.5, "temp_std": 3.0,
            "rain_mean": 260.0, "rain_std": 60.0,
            "humidity_mean": 55.0, "humidity_std": 8.0,
            "n_mean": 75.0, "n_std": 15.0,
            "p_mean": 42.0, "p_std": 9.0,
            "k_mean": 35.0, "k_std": 8.0,
            "ph_mean": 6.9, "ph_std": 0.4,
            "base_yield": 1.3, "yield_std": 0.35,
            "fertilizer_mean": 90.0, "pesticide_mean": 1.4
        },
        "Millets": {
            "seasons": ["Kharif"],
            "states": ["Rajasthan", "Maharashtra", "Karnataka", "Uttar Pradesh", "Gujarat", "Haryana"],
            "temp_mean": 29.0, "temp_std": 3.5,
            "rain_mean": 420.0, "rain_std": 95.0,
            "humidity_mean": 50.0, "humidity_std": 9.0,
            "n_mean": 50.0, "n_std": 12.0,
            "p_mean": 28.0, "p_std": 7.0,
            "k_mean": 25.0, "k_std": 7.0,
            "ph_mean": 7.2, "ph_std": 0.6,
            "base_yield": 1.4, "yield_std": 0.35,
            "fertilizer_mean": 60.0, "pesticide_mean": 0.8
        },
        "Soybean": {
            "seasons": ["Kharif"],
            "states": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Karnataka", "Telangana", "Gujarat"],
            "temp_mean": 26.0, "temp_std": 2.5,
            "rain_mean": 850.0, "rain_std": 150.0,
            "humidity_mean": 75.0, "humidity_std": 7.0,
            "n_mean": 40.0, "n_std": 10.0,
            "p_mean": 65.0, "p_std": 12.0,
            "k_mean": 45.0, "k_std": 10.0,
            "ph_mean": 6.6, "ph_std": 0.4,
            "base_yield": 1.8, "yield_std": 0.45,
            "fertilizer_mean": 85.0, "pesticide_mean": 1.6
        }
    }

    crops = list(crop_profiles.keys())
    rows_per_crop = num_rows // len(crops)

    data = []
    years = list(range(2005, 2024))

    for crop in crops:
        p = crop_profiles[crop]
        for _ in range(rows_per_crop):
            year = int(np.random.choice(years))
            season = np.random.choice(p["seasons"])
            state = np.random.choice(p["states"])

            # Soil
            n = max(5.0, np.random.normal(p["n_mean"], p["n_std"]))
            p_val = max(5.0, np.random.normal(p["p_mean"], p["p_std"]))
            k_val = max(5.0, np.random.normal(p["k_mean"], p["k_std"]))
            ph_val = np.clip(np.random.normal(p["ph_mean"], p["ph_std"]), 4.5, 9.0)

            # Weather
            temp = np.clip(np.random.normal(p["temp_mean"], p["temp_std"]), 8.0, 46.0)
            rain = max(50.0, np.random.normal(p["rain_mean"], p["rain_std"]))
            humidity = np.clip(np.random.normal(p["humidity_mean"], p["humidity_std"]), 20.0, 98.0)

            # Inputs
            fertilizer = max(10.0, np.random.normal(p["fertilizer_mean"], p["fertilizer_mean"] * 0.15))
            pesticide = max(0.1, np.random.normal(p["pesticide_mean"], p["pesticide_mean"] * 0.20))
            area = float(np.random.exponential(scale=1200.0) + 100.0)

            # Agronomic yield response calculation
            # Quadratic optimal response for rain, temp, ph
            temp_dev = ((temp - p["temp_mean"]) / p["temp_std"]) ** 2
            rain_dev = ((rain - p["rain_mean"]) / p["rain_std"]) ** 2
            ph_dev = ((ph_val - p["ph_mean"]) / p["ph_std"]) ** 2
            n_ratio = min(1.3, n / p["n_mean"])
            fert_ratio = min(1.3, fertilizer / p["fertilizer_mean"])

            penalty = 0.08 * temp_dev + 0.08 * rain_dev + 0.06 * ph_dev
            synergy = 0.05 * (n_ratio + fert_ratio)
            multiplier = max(0.35, 1.0 - penalty + synergy + np.random.normal(0, 0.08))

            yield_val = max(0.2, p["base_yield"] * multiplier)
            production = round(yield_val * area, 2)

            data.append({
                "Crop": crop,
                "Crop_Year": year,
                "Season": season,
                "State": state,
                "Area": round(area, 2),
                "Production": production,
                "Yield": round(yield_val, 2),
                "Fertilizer": round(fertilizer, 2),
                "Pesticide": round(pesticide, 2),
                "N": round(n, 2),
                "P": round(p_val, 2),
                "K": round(k_val, 2),
                "pH": round(ph_val, 2),
                "Temperature": round(temp, 2),
                "Rainfall": round(rain, 2),
                "Humidity": round(humidity, 2)
            })

    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic crop rows saved to: {output_path}")
    return df

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/SYNTHETIC_crop_yield.csv"
    generate_synthetic_crop_dataset(out)
