import json

def generate_unique_identifiers(file_path):
    # Dictionary to store the unique identifier and corresponding text
    unique_identifiers = {}

    with open(file_path, 'r') as file:
        # Reading each line in the file
        for line_number, line in enumerate(file, start=1):

            unique_identifiers[line_number] = line.strip()

    # Saving the result to a JSON file
    with open('corpus_id.json', 'w') as json_file:
        json.dump(unique_identifiers, json_file, indent=4)

    return "JSON file created successfully."

generate_unique_identifiers('corpus.txt') 


