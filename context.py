import os
import sys


def traverse_directory(directory, output_file):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    file_content = f.read()

                output_file.write(f"--- {file_path}\n")
                output_file.write(file_content)
                output_file.write("\n\n")


# Get the directory name from command line argument
if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    # If no directory is provided, concatenate all .txt files in the "temp" directory
    output_file_path = "out.txt"
    with open(output_file_path, "w") as output_file:
        temp_files = [file for file in os.listdir("temp")]
        for file in temp_files:
            file_path = os.path.join("temp", file)
            with open(file_path, "r") as input_file:
                file_content = input_file.read()
            output_file.write(file_content)
            output_file.write("\n")  # Add a newline between files
    print(f"Files concatenated successfully to {output_file_path}")
    sys.exit(0)

# Output file path based on directory name
output_file_path = f"temp/{directory}.txt"

# Open the output file in write mode
with open(output_file_path, "w") as output_file:
    # Traverse the directory and write to the output file
    traverse_directory(directory, output_file)

print(f"File paths and contents written to {output_file_path}")


# Check if an output file is specified as a command line argument
if len(sys.argv) > 2:
    output_file_path = sys.argv[2]
else:
    # If no output file is specified, concatenate all .txt files in the "temp" directory
    output_file_path = "out.txt"

# Open the output file in write mode
with open(output_file_path, "w") as output_file:
    # Get the list of .txt files in the "temp" directory
    temp_files = [file for file in os.listdir("temp") if file.endswith(".txt")]
    
    # Iterate over each .txt file in the "temp" directory
    for file in temp_files:
        file_path = os.path.join("temp", file)
        
        # Open the file and read its contents
        with open(file_path, "r") as input_file:
            file_content = input_file.read()
        
        # Write the file contents to the output file
        output_file.write(file_content)
        output_file.write("\n")  # Add a newline between files

print(f"Files concatenated successfully to {output_file_path}")
