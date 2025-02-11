import pandas as pd
import streamlit as st
import ast
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Constants for the dashboard
TITLE_ICON = "🏋️‍♂️"
HIDE_STREAMLIT_STYLE = """
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """


# Load the data with caching to improve performance
@st.cache_data
def get_data(file_path: str):
    """Loads data from a CSV file and returns a DataFrame."""
    df = pd.read_csv(file_path)

    # Ensure numeric columns have appropriate types
    df['Age'] = df['Age'].astype(int)
    df['Duration'] = df['Duration'].astype(int)
    df['BMI'] = df['BMI'].astype(float)
    df['Intensity'] = df['Intensity'].astype(int)

    # Apply One-Hot Encoding for the 'Gender' column
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    return df


def get_bmi_category(df):
    # Define the BMI categories based on ranges
    df['BMI Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, float('inf')],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'], right=False)

    return df


# Function to encode BMI categories using OneHotEncoder
def one_hot_encode_bmi_category(df):
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # Use sparse_output=False to get a dense array directly

    # Fit and transform the BMI Category column
    encoded_array = encoder.fit_transform(df[['BMI Category']])

    # Create a DataFrame for the encoded features
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(['BMI Category']),
        index=df.index  # Ensures alignment of indices between original df and encoded_df
    )

    # Concatenate the encoded features with the original DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    return df


def get_exercise_details(df, recommended_exercises):
    """
    Returns exercise types and image paths for the given list of recommended exercises.

    Parameters:
    - df: DataFrame containing exercise data with exercise types and image paths.
    - recommended_exercises: A list of recommended exercises.

    Returns:
    - exercise_details: A list of dictionaries containing exercise types and image paths for each recommended exercise.
    """
    # Initialize a list to hold the exercise details
    exercise_details = []

    # Iterate over each recommended exercise
    for exercise in recommended_exercises:
        # Filter the DataFrame for the current exercise
        exercise_info = df[df['Exercise'] == exercise]

        # If the exercise is found, extract the exercise types and image paths
        if not exercise_info.empty:
            for _, row in exercise_info.iterrows():
                exercise_details.append({
                    'Exercise Type': row['Exercise Type'],  # Assuming there's a column named 'Exercise Type'
                    'Image Path': row['Image Paths']  # Assuming there's a column named 'Image Paths'
                })

    return exercise_details


def flatten_image_paths(image_paths):
    """Flatten any nested lists of image paths."""
    if isinstance(image_paths, list):
        return [img for sublist in image_paths for img in (sublist if isinstance(sublist, list) else [sublist])]
    return []


def safe_literal_eval(value):
    try:
        return eval(value) if isinstance(value, str) else value
    except:
        return []


def combine_columns_per_category(df, exercise_category_column, exercise_types_columns_prefix,
                                 image_paths_columns_prefix):
    """Combine multiple columns for each exercise category."""
    combined_data = []

    for _, row in df.iterrows():
        exercise_category = row[exercise_category_column]

        # Combine all Exercise Types for the current exercise category
        exercise_types = []
        j = 0
        while f"{exercise_types_columns_prefix}{j}" in df.columns:
            exercise_types.append(row[f"{exercise_types_columns_prefix}{j}"])
            j += 1

        # Combine all Image Paths for the current exercise category
        image_paths = []
        j = 0
        while f"{image_paths_columns_prefix}{j}" in df.columns:
            image_paths.append(safe_literal_eval(row[f"{image_paths_columns_prefix}{j}"]))
            j += 1

        # Flatten the image paths
        image_paths = flatten_image_paths(image_paths)

        # Add combined exercise types and image paths to the data
        combined_data.append({
            'Exercise Category': exercise_category,
            'Combined Exercise Types': exercise_types,
            'Combined Image Paths': image_paths
        })

    # Convert combined data into a new DataFrame
    combined_df = pd.DataFrame(combined_data)
    return combined_df


# Function to recommend exercises using KNN
def knn_recommend_exercises(df, gender, age, bmi_category, intensity_category, duration):
    # Initialize the user profile with default values
    user_profile = {
        'Gender_Male': [1 if gender == 'Male' else 0],
        'Age': [age],
        'Intensity': [intensity_category],
        'Duration': [duration]
    }

    # Dynamically add BMI one-hot encoded columns to user profile
    for column in df.filter(like='BMI Category_').columns:
        user_profile[column] = [1 if column == f'BMI Category_{bmi_category}' else 0]

    # Convert user_profile to DataFrame
    user_profile_df = pd.DataFrame(user_profile)

    # Define feature columns for training
    feature_columns = ['Gender_Male', 'Age', 'Intensity', 'Duration'] + df.filter(like='BMI Category_').columns.tolist()

    # Prepare data for training
    X = df[feature_columns]
    y = df['Exercise']  # Assuming 'Exercise' is the target column

    # Standardize the data
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # Reindex the user profile to match the training feature columns
    user_profile_df = user_profile_df.reindex(columns=feature_columns, fill_value=0)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_std, y_train)

    # Standardize the user profile before prediction
    user_profile_std = sc.transform(user_profile_df)

    # Predict recommended exercises
    predicted_exercises = knn.predict(user_profile_std)

    # Evaluate model accuracy
    y_pred = knn.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)

    return predicted_exercises.tolist()


def clean_image_paths(image_paths):
    """Convert string paths to list and replace backslashes with forward slashes."""
    image_list = ast.literal_eval(image_paths)  # Convert string to actual list
    cleaned_paths = [path.replace("\\", "/") for path in image_list]
    return cleaned_paths


# Function to display exercise images based on the exercise category
def display_exercises(df):
    # Standardize column names to avoid issues with spaces or variations
    # Print the column names to confirm
    #st.write(f"Columns in the DataFrame: {df.columns}")
    #st.write(df.columns)

    exercise_columns = [col for col in df.columns if col.startswith('Exercise Types')]
    image_columns = [col for col in df.columns if col.startswith('Image Paths')]

    # Iterate over each row in the dataframe
    for idx, row in df.iterrows():
        exercise_category = row['Exercise_Category']
        st.write(f"### **Exercise Category: {exercise_category}**")

        # Iterate through the pairs of columns: Exercise Types and Image Paths
        for i, (exercise_col, image_col) in enumerate(zip(exercise_columns, image_columns)):
            exercise_type = row[exercise_col]
            image_path = row[image_col]

            # Check if both exercise type and image path exist for this pair
            if pd.notnull(exercise_type) and pd.notnull(image_path):
                st.write(f"**Exercise Type {i + 1}:** {exercise_type}")
                st.write(f"Processing Image {i + 1}: {image_path}")

                # Display the image if it exists
                if os.path.exists(image_path):
                    st.image(image_path, caption=exercise_type, use_column_width=True)
                else:
                    st.write(f"Image not found: {image_path}")
            else:
                st.write(f"Skipping pair {i + 1} (missing data).")


def streamlit_dashboard(df):
    """Set up the main dashboard for exercise recommendations."""

    # User input for filtering
    selected_gender = st.selectbox("Gender:", options=['Male', 'Female'])
    age = st.slider("Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].mean()))
    selected_bmi_category = st.selectbox("BMI Category:", options=df['BMI Category'].unique().tolist())
    selected_intensity_category = st.selectbox('Intensity:', options=df['Intensity'].unique().tolist())
    duration = st.slider("Duration", min_value=int(df['Duration'].min()), max_value=int(df['Duration'].max()),
                         value=int(df['Duration'].mean()))

    if st.button("Get Exercise Recommendation"):
        # Placeholder for exercise recommendation logic (replace with actual function)
        recommended_exercises = knn_recommend_exercises(df, selected_gender, age, selected_bmi_category,
                                                        selected_intensity_category, duration)

        # Debugging: Print recommended exercises
        st.write(f"Recommended Exercises: {recommended_exercises}")

        # Remove duplicates
        recommended_exercises = list(set(recommended_exercises))  # Remove duplicates

        if recommended_exercises:
            st.session_state['recommended_exercises'] = recommended_exercises
            st.write(
                f"Recommended Exercises for {selected_gender}, Age: {age}, BMI Category: {selected_bmi_category}, "
                f"Intensity: {selected_intensity_category}, Duration: {duration}:")
            for exercise in recommended_exercises:
                st.write(f"- {exercise}")
        else:
            st.write("No exercises found for the selected criteria.")

    if st.button("Display Recommended Exercises"):
        if 'recommended_exercises' in st.session_state:
            recommended_exercises = st.session_state.recommended_exercises

            if recommended_exercises:
                st.write("**Recommended Exercises Details:**")

                # Iterate through all recommended exercise categories
                for exercise_category in recommended_exercises:
                    # Filter the DataFrame for the specific exercise category
                    exercise_rows = df[df['Exercise Category'] == exercise_category]

                    if not exercise_rows.empty:
                        st.write(f"**Exercise Category:** {exercise_category}")

                        seen_exercises = set()  # Set to track seen exercises

                        for _, row in exercise_rows.iterrows():
                            # Loop through the columns that contain exercise types and image paths
                            for i in range(40):  # Assuming you have 40 columns (adjust based on your data)
                                exercise_col = f'Exercise Types.{i}' if i > 0 else 'Exercise Types'
                                image_col = f'Image Paths.{i}' if i > 0 else 'Image Paths'

                                # Check if the exercise type is not empty
                                exercise_type = row.get(exercise_col)
                                if pd.notna(exercise_type) and exercise_type not in seen_exercises:
                                    st.write(f"- **Exercise Type:** {exercise_type}")
                                    seen_exercises.add(exercise_type)  # Add the exercise to the set

                                    image_paths = safe_literal_eval(row.get(image_col))  # Get image paths safely

                                    # Flatten the image paths if needed
                                    image_paths = flatten_image_paths(image_paths)

                                    for img in image_paths:
                                        # Ensure the correct file path
                                        img = img.replace("\\", "/")  # Normalize the file path for Windows

                                        # Check if the image exists
                                        if os.path.exists(img):
                                            st.image(img, caption=exercise_type, use_column_width=True)
                                        else:
                                            st.write(f"Image not found: {img}")
                                elif exercise_type in seen_exercises:
                                    # Skip the duplicate exercise type
                                    continue
                    else:
                        st.write(f"No details found for the exercise category: {exercise_category}.")
            else:
                st.write("No recommended exercises found.")
        else:
            st.write("Please get recommendations first before displaying exercises.")


def app():
    st.markdown(
        f"""
        <h1 style='text-align: center; color: black;'> {TITLE_ICON} Revivo Basic Recommendation</h1>""",
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
            <h1 style='text-align: left; font-size: 20px; color: black;'> Exercise Recommendation Dashboard </h1>
            """,
        unsafe_allow_html=True
    )
    st.markdown("""This dashboard provides key insights into exercise recommendations based on user profiles.
    You can filter recommendations based on your personal characteristics, such as age, gender, BMI, and exercise intensity.
    Explore and discover exercises that best match your fitness goals!
    """)
    st.divider()

    file_path = "C:/Users/ERC/Desktop/cleaned_exercise_data.csv"

    # Load and preprocess the data
    df = get_data(file_path)

    df = get_bmi_category(df)  # Add BMI category
    df = one_hot_encode_bmi_category(df)  # Encode BMI
    streamlit_dashboard(df)
    # Input fields


if __name__ == "__main__":
    app()
