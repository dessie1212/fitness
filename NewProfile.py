import streamlit as st


TITLE_ICON = "🧑‍💻"
HIDE_STREAMLIT_STYLE = """
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """


def streamlit_dashboard():
    """Revivo Wellness Center: User Input Page."""
    # Title and description
    st.markdown("""
           <style>
           /* Set a custom background color */
           .css-1aumxhk {
               background-color: #F5F1E3 !important;
           </style>
       """, unsafe_allow_html=True)

    st.markdown(
        f"""
            <h1 style='text-align: center; color: black;'> {TITLE_ICON} Revivo Wellness Center {TITLE_ICON} </h1>
            """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
                    <h3 style='text-align: center; font-size: 20px; color: black;'> Welcome to Our Center! Enter your details. </h1>
                    """,
        unsafe_allow_html=True
    )


    st.divider()
    actual_weight = st.number_input("Actual Weight (per pound)", value=0.0, min_value=0.0, step=0.1, format="%.1f")
    dream_weight = st.number_input("Enter Dream Weight (per pound):", value=0.0, min_value=0.0, step=0.1, format="%.1f")
    feet = st.number_input("Enter Height (Feet):", min_value=0, max_value=9, value=5, step=1)
    inches = st.number_input("Enter Height (Inches):", min_value=0, max_value=11, value=6, step=1)
    age = st.slider("Age (years)", min_value=18, max_value=100, value=18, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    bmi = st.number_input("Enter your BMI", value=18.0, min_value=0.0, step=0.1, format="%.1f")
    duration = st.slider("Preferred Workout Duration (minutes)", value=0, min_value=0, max_value=120, step=5)
    intensity = st.number_input("Preferred Exercise Intensity", min_value=1, max_value=11, value=1, step=1)


    st.header("User Summary")
    st.write(f"- **Dream Weight:** {dream_weight} pound")
    st.write(f"- **Actual Weight:** {actual_weight} pound")
    st.write(f"- **Height:** {feet} feet {inches} inches")
    st.write(f"- **Age:** {age} years")
    st.write(f"- **Gender:** {gender}")
    st.write(f"- **BMI:** {bmi}")
    st.write(f"- **Preferred Duration:** {duration} minutes")
    st.write(f"- **Preferred Exercise Intensity:** {intensity}")

    if st.button("Send personal information to Kafka"):
        # Display success message when button is pressed
        st.success("Your details have been successfully updated!")

    st.markdown("""
            <style>
            body {
                background-color: #F5F1E3 !important;  /* Light nude/beige background */
            }
            </style>
        """, unsafe_allow_html=True)

def app():

    streamlit_dashboard()

# To test this file independently
if __name__ == "__main__":
    app()


