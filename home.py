import streamlit as st
"""
# Constants for the dashboard
TITLE_ICON = ":house:"
HIDE_STREAMLIT_STYLE = 
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    
"""
# Introduction section 
def display_introduction():
    st.markdown(
        """
        <h1 style='text-align: center; color: black;'> 💪 Revivo Wellness Center Dashboard </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
                        <h3 style='text-align: left; font-size: 20px; color: black;'> Welcome To Your Personalized Exercise Recommendation System! </h1>
                        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    
    This dashboard is designed to provide you with customized exercise recommendations based on your personal profile. It utilizes machine learning models to help you find the best exercises suited to your needs.
    """)

    st.markdown(
        f"""
                        <h3 style='text-align: left; font-size: 20px; color: black;'> Explore the Dashboard: </h1>
                        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    - **Home:** Learn about the application and get an overview of its functionalities.            
    - **Basic Recommendation:** Get exercise recommendation based on your profile.
    - **New User Profile:** Create your personalized user profile by entering key details such as weight, height, age, etc. 
    Based on this information, help feed our machine learning-based recommendation system to receive tailored exercise recommendations that align with your fitness goals. 
     """)

    st.markdown(
        f"""
                            <h3 style='text-align: left; font-size: 20px; color: black;'> How to Use This Dashboard </h1>
                            """,
        unsafe_allow_html=True
    )
    st.markdown("""
    1. **Select a Section:** Use the sidebar menu to navigate between different sections.
    2. **Provide Your Input:** Depending on the selected section, provide your profile data like Age, Gender, BMI, etc.
    3. **View Recommendations:** Get personalized exercise recommendations, or share your user profile.

    **Let's get started!** Use the sidebar to navigate through the options.
    """)


# Main function to set up the dashboard
def app():
    """
    st.title(f"{TITLE_ICON} Home")
    """
    # Display introduction on the Home page
    display_introduction()

    st.divider()
