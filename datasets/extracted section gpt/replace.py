import csv
import os

# Load the CSV data into a dictionary
def load_replacements(csv_path):
    replacements = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile,delimiter=";")
        for row in reader:
            if len(row) < 2:
                # Skip rows that don't have at least two columns
                continue
            key = row[0].strip()
            value = row[1].strip()
            replacements[key] = value
    return replacements

# Process the files in the folders
def process_files(folders, replacements):
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                with open(filepath, 'w', encoding='utf-8') as file:
                    for line in lines:
                        if line.startswith('- '):
                            search_string = line[2:].strip()  # Get the string after "- "
                            print(search_string)
                            if search_string in replacements:
                                # Replace the line
                                file.write(f"- {replacements[search_string]}\n")
                            # If no match is found, don't write the line (effectively deleting it)
                        else:
                            # Write other lines as they are
                            file.write(line)

# Define the folders to process
folders_to_process = ['bmj', 'nejm', 'jama', 'lancet']

# Load replacements from CSV
replacements_dict = load_replacements('all_filtered_techniques_simplify.csv')

# Process the files
process_files(folders_to_process, replacements_dict)
