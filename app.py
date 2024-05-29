import streamlit as st
from streamlit_option_menu import option_menu
from app_page1 import introduction_page
from app_page2 import about_the_dataset
from app_page3 import heart_disease_prediction_page


def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
    
    # Sidebar menu
    with st.sidebar:
        selected = option_menu(
            menu_title=None, 
            options=["Introduction", "Data Exploration & Machine Learning Models", "Heart Disease Prediction"], 
            icons=['house', 'database', 'heart'], 
            menu_icon="cast", 
            default_index=0)
        
        st.markdown("### Connect with me")
        
        # Adding icons for GitHub and LinkedIn
        github_url = "https://github.com/jessih828"
        linkedin_url = "https://www.linkedin.com/in/hsieh-jessica/"

        # URLs for the icons
        github_icon = "https://cdn-icons-png.flaticon.com/512/25/25231.png"
        linkedin_icon = "https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg"
        
        # Display icons with adjusted size using HTML
        st.markdown(f'<a href="{github_url}" target="_blank"><img src="{github_icon}" width="30" height="30"></a>', unsafe_allow_html=True)
        st.markdown(f'<a href="{linkedin_url}" target="_blank"><img src="{linkedin_icon}" width="30" height="30"></a>', unsafe_allow_html=True)

    # Load the selected page
    if selected == "Introduction":
        introduction_page()
    elif selected == "Data Exploration & Machine Learning Models":
        about_the_dataset()
    elif selected == "Heart Disease Prediction":
        heart_disease_prediction_page()

if __name__ == "__main__":
    main()