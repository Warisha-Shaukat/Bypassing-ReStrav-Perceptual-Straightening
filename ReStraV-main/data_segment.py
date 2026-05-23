import shutil
import os
from pathlib import Path
import random

# --- CONFIGURATION ---
# Source folders
fake_src = Path("test_data/fake")
real_src = Path("test_data/real")

# Output folders
output_combined = Path("testog")     # 50 Real + 50 Fake
output_real = Path("test_real")      # 50 Real only
output_fake = Path("test_fake")      # 50 Fake only

count_per_class = 500

def process_segmentation():
    """
    Samples 50 videos from test_data/fake and 50 videos from testdata/real.
    Copies them to:
    - 'testog' (Combined)
    - 'test_real' or 'test_fake' (Specific)
    """
    print(f"Starting segmentation process...")
    
    # 1. Ensure all output directories exist
    for folder in [output_combined, output_real, output_fake]:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Ensured folder exists: {folder}")
    
    # 2. Define sources and their specific output targets
    sources = [
        (fake_src, output_fake, "FAKE"),
        (real_src, output_real, "REAL")
    ]
    
    total_copied = 0
    
    # 3. Process each source
    for src_path, specific_dest, label in sources:
        print(f"\nAccessing {label} videos at: {src_path}")
        
        if not src_path.exists():
            print(f"Error: Source folder '{src_path}' not found.")
            continue
            
        # Get list of files
        all_files = [f for f in src_path.iterdir() if f.is_file()]
        num_available = len(all_files)
        
        if num_available == 0:
            print(f"Warning: No files found in {src_path}.")
            continue
            
        # Determine how many to take
        num_to_take = min(count_per_class, num_available)
        if num_available < count_per_class:
            print(f"Warning: Only {num_available} files available. Taking all of them.")
            
        # Randomly sample the files
        selected_files = random.sample(all_files, num_to_take)
        
        # Copy files to BOTH locations
        for i, f in enumerate(selected_files):
            try:
                # 1. Copy to combined folder (testog)
                shutil.copy2(f, output_combined / f.name)
                
                # 2. Copy to specific class folder (test_real or test_fake)
                shutil.copy2(f, specific_dest / f.name)
                
                total_copied += 1
            except Exception as e:
                print(f"Error copying {f.name}: {e}")
                
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_to_take} from {label}...")

    print(f"\n--- Summary ---")
    print(f"Total unique videos sampled: {total_copied}")
    print(f"Combined folder '{output_combined}' populated.")
    print(f"Specific folders '{output_real}' and '{output_fake}' populated.")
    print("Process complete.")

if __name__ == "__main__":
    process_segmentation()