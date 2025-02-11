import pandas as pd

# Load the Excel file
file_path = "C:/Users/ERC/Desktop/combined_exercise_and_links2.csv"
df = pd.read_csv(file_path)

# Check the names of the columns to ensure that 'Exercise Types.31' to 'Exercise Types.40' are present
df.rename(columns={
    'Image Paths ': 'Image Paths.31',  # Fixing the extra space
    'Images Paths': 'Image Paths.32',  # Renaming to match the correct structure
    'Images Paths.1': 'Image Paths.33',  # Renaming to match the correct structure
    'Images Paths.2': 'Image Paths.34',
    'Images Paths.3': 'Image Paths.35',
    'Images Paths.4': 'Image Paths.36',
    'Images Paths.5': 'Image Paths.37',
    'Images Paths.6': 'Image Paths.38',
    'Images Paths.7': 'Image Paths.39',
    'Images Paths.8': 'Image Paths.40'
}, inplace=True)

# Check the columns again after renaming
print(df.columns)

# Now, you can inspect the columns from Exercise Types.31 to Exercise Types.40
exercise_types_columns = df.loc[:, 'Exercise Types.31':'Exercise Types.40']
print(exercise_types_columns.head())
# Extract and display the relevant columns (from 'Exercise Types.31' to 'Exercise Types.40')
exercise_types_columns = df.loc[:, 'Exercise Types.31':'Exercise Types.40']
print(exercise_types_columns.head())  # Display the first few rows of these columns
output_file_path = r"C:\Users\ERC\Desktop\Assignment 6\cleaned_exercise_data.csv"

# Save the DataFrame to the specified path
df.to_csv(output_file_path, index=False)

print(f"Data has been successfully saved to {output_file_path}")