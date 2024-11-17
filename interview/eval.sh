#!/bin/bash

DIR="./interview/harness_eval"

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    usage
fi

# Counter for scripts
total_scripts=0
successful_scripts=0
failed_scripts=0

echo "Starting to execute bash scripts in $DIR"
echo "----------------------------------------"

# Find and execute all .sh files
for script in "$DIR"/*.sh; do
    # Skip if no files found
    if [ ! -e "$script" ]; then
        echo "No bash scripts (.sh files) found in $DIR"
        exit 0
    fi
    
    # Skip the current script to prevent infinite loop
    if [ "$(basename "$script")" = "$(basename "$0")" ]; then
        continue
    fi
    
    total_scripts=$((total_scripts + 1))
    echo "Executing: $script"
    
    # Execute script using bash directly
    if bash "$script"; then
        echo "✓ Success: $script"
        successful_scripts=$((successful_scripts + 1))
    else
        echo "✗ Failed: $script"
        failed_scripts=$((failed_scripts + 1))
    fi
    echo "----------------------------------------"
done

# Print summary
echo "Execution Summary:"
echo "Total scripts: $total_scripts"
echo "Successful: $successful_scripts"
echo "Failed: $failed_scripts"
