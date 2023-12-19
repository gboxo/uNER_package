import os

# Paths for the source folders
source_folders = [
    'jama',
    'bmj',
    'lancet',
    'nejm'
]

# Path for the new text file
output_file_path = 'corpus.txt'

# Open the new file in write mode
with open(output_file_path, 'w') as output_file:
    # Loop through each source folder
    for folder in source_folders:
        # Ensure the folder exists
        if os.path.exists(folder):
            # List all files in the folder
            for filename in os.listdir(folder):
                # Check if the file is a text file
                if filename.endswith('.txt'):
                    # Create the complete path for the file
                    file_path = os.path.join(folder, filename)
                    
                    # Read the file's content
                    with open(file_path, 'r') as source_file:
                        content = source_file.read()
                        
                        # Replace '\n' with ' ' and write to the new file

                        output_file.write(content )
        else:
            print(f"Folder {folder} doesn't exist!")

print("Process completed!")
