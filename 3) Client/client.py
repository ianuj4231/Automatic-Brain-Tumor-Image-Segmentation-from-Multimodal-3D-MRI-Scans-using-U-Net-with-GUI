import streamlit as st
import subprocess
import os 
from PIL import Image

os.chdir("D:/pfm/brats2020/clientout/server")

def run_brats2019():
    subprocess.run(["streamlit", "run", "brats2019_backend.py"], shell=True)

def run_brats2020():
    subprocess.run(["streamlit", "run", "brats2020_backend.py"], shell=True)

def run_brats2021():
    subprocess.run(["streamlit", "run", "brats2021_backend.py"], shell=True)

def main():
    st.set_page_config(page_title="Web App", page_icon=":brain:", layout="wide")

    html = """
        <div style='border: 8px solid #000073; 
                    box-shadow: 0 0 34px #000073, 0 0 34px #000073, 0 0 34px #000073, 0 0 34px #000073;'>
        <div style='text-align: center; background-color: black; padding: 20px;'>
            <p style='font-family: Verdana; margin:0; font-size: 39px; font-weight: bold;
                      color: #FFFF00;'> Automatic Brain Tumor Segmentation from Multi-Modal 3D MRI Scans using U-Net </p>
        </div>
        </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

    st.write(" ") 
    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.write(" ")

    col1, col2 = st.columns([55,45])
    
    with col1:
        image = Image.open("doctors.png")
        st.image(image, width=970)            
        
    with col2:
        st.markdown(
            """
            <div style='background-color:#000000; 
            padding:20px; border: 5px solid #000073;
            box-shadow: 0 0 25px #000073, 0 0 25px #000073, 0 0  25px #000073, 0 0 25px #000073;
            '>
                <p style='font-weight:bold; 
                color:#039aff ; font-size:33px; font-family:Verdana;'>
                Hello, Doctor. Please select a dataset you want to work with:
                </p>
                <div style='margin-top:20px;'>
                    
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;font-size: 28px;}</style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;font-size: 28px;}</style>', unsafe_allow_html=True)
        st.write(" ")
        st.write(" ")
        st.write(" ")
                
        dataset = st.radio("", ("BraTS 2019", "BraTS 2020", "BraTS 2021"), index=0)
        st.write(" ")

        st.markdown("""
            <style>
            div.stButton > button:first-child {
            background-color: #0000ff;
            margin: 0 auto;
            font-size: 50px !important;
            font-weight:bold;
            font-family:Verdana;
            
            color: #ffffff;
            display: block;
            width: 270px;
            height: 80px;
            border: 2px solid #0000ff;
            border-color: #0000ff;
            text-shadow: 4px 4px 8px rgba(0, 0, 0, 0.5);
            box-sizing: border-box;
            }
            div.stButton > button:hover {
                background-color: #0000ff;
                margin: 0 auto;
                font-family:Verdana;
                border-color: #0000ff;
                color: #ffffff;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); 
                font-weight:bold;
                display: block;
                width: 290px;
                height: 85px;
                border: 3px solid #0000ff;
                box-sizing: border-box;
                font-size: 40px;
            }
            """, unsafe_allow_html=True) 
        st.write(" ")
        st.write(" ")


        if st.button("Confirm Dataset"):
          
            if dataset == "BraTS 2019":
                run_brats2019()
            elif dataset == "BraTS 2020":
                run_brats2020()
            elif dataset == "BraTS 2021":
                run_brats2021()

if __name__ == "__main__":
    main()