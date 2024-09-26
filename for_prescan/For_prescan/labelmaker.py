import csv
import os

def create_labels_csv(filename='ACC_Scenario_006_12_5_labels.csv', total_entries=173, start_ones=164, end_ones=173):
    """
    Create a CSV file with labels.
    
    Args:
    filename (str): Name of the output CSV file
    total_entries (int): Total number of entries
    start_ones (int): Start index for entries labeled as 1 (inclusive)
    end_ones (int): End index for entries labeled as 1 (inclusive)
    """
    
    # Create the labels
    labels = [0] * total_entries
    for i in range(start_ones - 1, end_ones):  # -1 because CSV rows are 1-indexed but Python is 0-indexed
        labels[i] = 1
    
    # Write to CSV
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for label in labels:
            writer.writerow([label])
    
    print(f"CSV file '{filename}' has been created with {total_entries} entries.")
    print(f"Entries {start_ones} to {end_ones} are labeled as 1, the rest are 0.")

# Create the CSV file
create_labels_csv()