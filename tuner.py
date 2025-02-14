import os
import subprocess
import pandas as pd

# Define the range of alpha and beta values to test
alpha_values = [0.05, 0.1, 0.2, 0.5, 1.0]  # Adjust as needed
beta_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Adjust as needed

# Path to your config file
config_path = "config.py"

# Store results
results = []

def update_config(alpha, beta):
    """
    Modify only 'alpha' and 'beta' in 'config.py' without deleting other settings.
    """
    config_template = f"""
# config.py

# 1) NO SMOOTH
no_smoothing = {{
    "method_name": "NO_SMOOTH",
    "n": 2
}}

# 2) ADD-K
add_k = {{
    "method_name": "ADD_K",
    "n": 2,
    "k": 1.0
}}

# 3) STUPID BACKOFF
stupid_backoff = {{
    "method_name": "STUPID_BACKOFF",
    "n": 9,   
    "alpha": 0.4
}}

# 4) GOOD TURING
good_turing = {{
    "method_name": "GOOD_TURING",
    "n": 2
}}

# 5) INTERPOLATION
interpolation = {{
    "method_name": "INTERPOLATION",
    "n": 2,
    "lambdas": [0.95, 0.5]
}}

# 6) KNESER-NEY
kneser_ney = {{
    "method_name": "KNESER_NEY",
    "n": 3,
    "discount": 0.75
}}

# 7) ERROR CORRECTION
error_correction = {{
    "internal_ngram_best_config": {{
        "method_name": "STUPID_BACKOFF",
        "n": 9,   
        "alpha": 0.4
    }},
    "candidate_max_distance": 3,
    "alpha": {alpha},  # weight for normalized edit distance
    "beta": {beta}     # weight for negative log n-gram probability
}}
"""
    with open("config.py", "w") as f:
        f.write(config_template)

    with open(config_path, "w") as f:
        f.write(config_template)

for alpha in alpha_values:
    for beta in beta_values:
        # Update config.py with new parameters
        update_config(alpha, beta)

        print(f"Running with alpha={alpha}, beta={beta}")

        # Run the grader.py script and capture output
        try:
            result = subprocess.run(
            ["python", "grader.py"],
            capture_output=True,
            text=True,
            check=False  # Do not raise exception immediately
)


            output = result.stdout
            score_line = [line for line in output.split("\n") if "Score:" in line]

            if score_line:
                score = float(score_line[0].split(":")[-1].strip())
            else:
                score = None

            print(f"Score: {score}")

            # Store results
            results.append({"alpha": alpha, "beta": beta, "score": score})

        except subprocess.CalledProcessError as e:
            print(f"Error running grader.py for alpha={alpha}, beta={beta}")
            print(e.output)
            results.append({"alpha": alpha, "beta": beta, "score": None})

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("results.csv", index=False)

print("Tuning complete! Results saved to results.csv.")
