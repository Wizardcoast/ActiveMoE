import os
import mmap

# Replace with your file path
file_path = 'your_file_here'

# Get the size of the file
file_size = os.path.getsize(file_path)

# Your desired offset
offset = 1000000  # Replace with the offset you want to check

# Check if the offset is greater than the file size
if offset > file_size:
    print(f"Offset {offset} is larger than the file size {file_size}.")
else:
    print(f"Offset {offset} is within the file size {file_size}.")

    # Open the file
    with open(file_path, 'r+b') as file:
        # Create a memory-mapped file
        mm = mmap.mmap(file.fileno(), length=0, offset=offset, access=mmap.ACCESS_WRITE)

        # ... Do something with the memory-mapped file

        # Close the memory map
        mm.close()

