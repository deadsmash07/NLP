#!/usr/bin/env python3
import os
import sys
import importlib
import pandas as pd
from grader import Grader

# Path to the config file
CONFIG_PATH = "config.py"

def update_config(method_name, n, **kwargs):
    """
    Modify 'config.py' to update specific n-gram parameters dynamically
    without modifying unrelated parameters.
    
    :param method_name: N-gram method name (e.g., "STUPID_BACKOFF")
    :param n: N-gram size (integer)
    :param kwargs: Additional parameters like alpha, beta, discount, etc.
    """
    try:
        # Read the existing config file
        with open(CONFIG_PATH, "r") as f:
            config_lines = f.readlines()
        
        updated_lines = []
        inside_target_section = False

        for line in config_lines:
            # Identify the section we need to update
            if f'"method_name": "{method_name}"' in line:
                inside_target_section = True

            # Identify when we exit the section
            if inside_target_section and line.strip() == "}":
                inside_target_section = False

            # Modify only relevant lines
            for key, value in kwargs.items():
                if inside_target_section and f'"{key}":' in line:
                    line = f'    "{key}": {value},\n'

            updated_lines.append(line)

        # Write back the modified config
        with open(CONFIG_PATH, "w") as f:
            f.writelines(updated_lines)

        print(f"[INFO] Successfully updated config.py with: {method_name}, n={n}, params={kwargs}")

    except Exception as e:
        print(f"[ERROR] Failed to update config.py: {e}")

def reload_config():
    """
    Force reloads the config module to ensure new settings are applied.
    """
    try:
        if "config" in sys.modules:
            del sys.modules["config"]
        import config
        importlib.reload(config)
        print("[INFO] Config successfully reloaded")
        return config
    except Exception as e:
        print(f"[ERROR] Config reload failed: {e}")
        return None

def tune_ngram(method_name, param_grid, param_names, file_name):
    """
    Generic function to tune n-gram model parameters.
    
    :param method_name: Name of the smoothing method (e.g., "STUPID_BACKOFF")
    :param param_grid: List of parameter combinations to try
    :param param_names: Corresponding parameter names (list)
    :param file_name: CSV file to save results
    """
    best_score = -1.0
    best_params = {}
    results = []

    for params in param_grid:
        # Extract primary parameter (n-gram size) and other params dynamically
        n = params[0]
        extra_params = {param_names[i]: params[i] for i in range(1, len(params))}

        # Update config.py before running the grader
        update_config(method_name, n, **extra_params)

        # Reload the config module
        config = reload_config()
        if config is None:
            print("[ERROR] Skipping this run due to config reload failure")
            continue

        # Run the grader
        print(f"\n=== Running {method_name} with {dict(zip(param_names, params))} ===")
        print(f"Updated Config: {config.error_correction}")

        try:
            grader_instance = Grader()
            score = grader_instance.grade(test_mode=False)  # Run the grading function
            print(f"Score: {score}")

            # Save results
            result_entry = {param_names[i]: params[i] for i in range(len(params))}
            result_entry["method"] = method_name
            result_entry["score"] = score
            results.append(result_entry)

            # Track best score
            if score > best_score:
                best_score = score
                best_params = result_entry.copy()
        
        except Exception as e:
            print(f"[ERROR] Grading failed for {params}: {e}")
            continue

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("./tuning_results", exist_ok=True)
    results_df.to_csv(f"./tuning_results/{file_name}.csv", index=False)
    
    print(f"Best {method_name} parameters:", best_params)

def main():
    # Define tuning parameter ranges
    stupid_backoff_grid = [
        (n, alpha) for n in [3, 4, 5, 6] for alpha in [0.2, 0.3, 0.4, 0.5, 0.6]
    ]
    interpolation_grid = [
        (2, lambdas) for lambdas in [[0.95, 0.05], [0.90, 0.10], [0.80, 0.20], [0.70, 0.30]]
    ]
    kneser_ney_grid = [
        (3, discount) for discount in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
    ]

    # Run tuning with updated config writing
    tune_ngram("STUPID_BACKOFF", stupid_backoff_grid, ["n", "alpha"], "stupid_backoff_tuning")
    tune_ngram("INTERPOLATION", interpolation_grid, ["n", "lambdas"], "interpolation_tuning")
    tune_ngram("KNESER_NEY", kneser_ney_grid, ["n", "discount"], "kneser_ney_tuning")

if __name__ == "__main__":
    main()
