import os


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


# Directory to traverse
directory = "mistral-src"

# Output file path
output_file_path = "output.txt"

# Open the output file in write mode
with open(output_file_path, "w") as output_file:
    # Traverse the directory and write to the output file
    traverse_directory(directory, output_file)

print(f"File paths and contents written to {output_file_path}")
