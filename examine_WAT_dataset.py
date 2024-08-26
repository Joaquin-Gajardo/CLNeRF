#%%
from collections import defaultdict
from pathlib import Path

import pandas as pd

#%%
path = Path('dataset/WAT')

#%%
scenes = list(path.iterdir())
scenes

# %%
def analyze_dataset(root_path):
    root = Path(root_path)
    summary = defaultdict(lambda: {"video_count": 0, "total_images": 0})
    
    for scene in scenes:
        if scene.is_dir() and scene.name != ".DS_Store":
            images_folder = scene / "images"
            timestep_paths = [timestep for timestep in images_folder.iterdir()]
            summary[scene.name]["video_count"] = len(timestep_paths)
            for image_folder in timestep_paths:
                summary[scene.name]["total_images"] += len(list(image_folder.glob("*.png")))

    # Calculate averages and prepare data for DataFrame
    data = []
    for scene, stats in summary.items():
        avg_images = stats["total_images"] / stats["video_count"] if stats["video_count"] > 0 else 0
        data.append({
            "Scene": scene,
            "Video count": stats["video_count"],
            "Avg images per video": int(round(avg_images, 2))
        })
    
    # Create and return the DataFrame
    return pd.DataFrame(data).sort_values("Scene")

#%%
# Use the function
df = analyze_dataset("dataset/WAT")
print(df.to_string(index=False))


# %%

import json
import os

import pandas as pd


def summarize_results(base_folder):
    # List to store results
    results = []

    # Iterate through each subfolder in the base folder
    for scene_folder in os.listdir(base_folder):
        scene_path = os.path.join(base_folder, scene_folder)
        
        # Check if it's a directory
        if os.path.isdir(scene_path):
            json_path = os.path.join(scene_path, 'test_results.json')
            
            # Check if the JSON file exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract mean values
                result = {
                    'Scene': scene_folder,
                    'PSNR': data.get('mean_psnr', 'N/A'),
                    'SSIM': data.get('mean_ssim', 'N/A'),
                    'LPIPS': data.get('mean_lpips', 'N/A')
                }
                results.append(result)
            else:
                print(f"Warning: No JSON file found for scene {scene_folder}")

    # Create a DataFrame
    df = pd.DataFrame(results)

    # Calculate overall mean
    mean_row = pd.DataFrame({
        'Scene': ['Overall Mean'],
        'PSNR': [df['PSNR'].mean()],
        'SSIM': [df['SSIM'].mean()],
        'LPIPS': [df['LPIPS'].mean()]
    })

    # Concatenate the mean row to the original DataFrame
    df = pd.concat([df, mean_row], ignore_index=True)

    # Set 'Scene' as index for better display
    df.set_index('Scene', inplace=True)

    return df

#%%
# Usage
base_folder = "results/colmap_ngpa"
summary_table = summarize_results(base_folder)

# Print the table
print(summary_table.to_string())

# Optionally, save to CSV
summary_table.to_csv("CLNeRF_UB_WAT_results_summary.csv")

# %%
