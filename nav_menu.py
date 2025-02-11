import streamlit as st
from streamlit_option_menu import option_menu


# Sidebar navigation
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })



    def run(self):
        # Sidebar navigation
        with st.sidebar:
            selected_app = option_menu(
                menu_title="Navigation Menu",
                options=["Home", "Basic Recommendation", "New User Profile"],
                icons=["house-fill", "trophy-fill", "person"],
                menu_icon="cast",
                default_index=0,
                orientation="vertical",
                styles={
                    # Sidebar container background color
                    "container": {
                        "padding": "5!important",
                        "background-color": "#808080",
                    },
                # Icon colors and size
                    "icon": {
                        "color": "#FFD700",  # Gold color for icons
                        "font-size": "20px",
                    },
                    # Unselected nav-link styling
                    "nav-link": {
                        "color": "#FFFFFF",
                        "font-size": "18px",
                        "text-align": "center",
                        "margin": "5px",
                        "border-radius": "5px",
                    },
                    # Selected nav-link styling
                    "nav-link-selected": {
                        "background-color": "#F5F5DC",
                        "color": "black",
                        "font-weight": "bold",
                    },
                }
            )

        # Run the selected application
        for app in self.apps:
            if app['title'] == selected_app:
                app['function']()

