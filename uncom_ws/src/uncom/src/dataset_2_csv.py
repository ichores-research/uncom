import os
import csv
import sys

def get_mp4_files(folder_path):
    # Get list of all files in the folder
    files = os.listdir(folder_path)
    # Filter out only .mp4 files
    mp4_files = [file for file in files if file.endswith('.mp4')]
    # Sort the list in alphanumeric order
    mp4_files.sort()
    print(mp4_files)

    return mp4_files

def save_to_csv(file_names, output_file):
    # Write the file names to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name'])
        for name in file_names:
            writer.writerow([name])

if __name__ == "__main__":
    # Get the folder path from CLI argument
    folder_path = sys.argv[1]
    # Get the list of .mp4 files
    mp4_files = get_mp4_files(folder_path)
    # Save the list to a CSV file
    save_to_csv(mp4_files, './mp4_files.csv')