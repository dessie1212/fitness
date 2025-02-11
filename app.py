import streamlit as st
from nav_menu import MultiApp
import home, basic_reco, NewProfile
# Constants for styling and page configuration
PAGE_ICON = ":bar_chart:"
LAYOUT = "wide"

def apply_background_color():
    st.markdown("""
            <style>
            {
            }
            </style>
        """, unsafe_allow_html=True)


# Main function to run the app
def main():
    # Configuration of Streamlit page

    st.set_page_config(
        page_title="Personalized Exercise Recommendations",
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )

    apply_background_color()
    # Define the applications
    app = MultiApp()

    # Add the different pages to the MultiApp
    app.add_app("Home", home.app)
    app.add_app("Basic Recommendation", basic_reco.app)
    app.add_app("New User Profile", NewProfile.app)

    app.run()

if __name__ == "__main__":
    main()
