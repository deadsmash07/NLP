#!/usr/bin/env python3
import sys
import importlib
import config
from grader import Grader

def reload_config():
    """Forces config reload to ensure changes are applied."""
    importlib.reload(config)
    return config

def main():
    if len(sys.argv) < 3:
        print("Usage: python grader_runner.py <alpha> <beta>")
        sys.exit(1)

    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])

    # Reload config before modifying
    config = reload_config()

    # Update config values
    config.error_correction["alpha"] = alpha
    config.error_correction["beta"] = beta

    print(f"Updated Config: {config.error_correction}")

    try:
        # Run Grader
        grader_instance = Grader()
        score = grader_instance.grade(test_mode=False)

        if score is None:
            print("ERROR: Grader returned None as score!", file=sys.stderr)
            sys.exit(1)

        print(f"Score: {score}")

    except Exception as e:
        print(f"ERROR: Exception occurred while grading - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
