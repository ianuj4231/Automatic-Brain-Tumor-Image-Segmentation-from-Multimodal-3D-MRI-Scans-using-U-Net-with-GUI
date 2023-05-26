import os
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
import streamlit as st
from utils import preprocess, predict_function, plot_slices
st.set_page_config(page_title="Brats2020_WebPage", page_icon=":brain:", layout="wide")


def main():
    fig = None
    st.markdown(
        """
        <div style='background-color:#000000; 
        padding:20px; border: 5px solid #000073;
        box-shadow: 0 0 25px #000073, 0 0 25px #000073, 0 0  25px #000073, 0 0 25px #000073;
        '>
            <p style='font-weight:bold; 
            color:#FFFF00  ; font-size:36px; font-family:Verdana; text-align:center'>
            You are currently working on the Brats2020 dataset.
            </p>
            <div style='margin-top:20px;'>
                
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
    col1, col2, col3= st.columns([18,0.8,19])
    with col1:
        st.write(" ")
        st.write(" ")
        
        st.markdown(
            """
            <div style='background-color:#000000; 
            padding:20px; border: 5px solid #000073;
            box-shadow: 0 0 25px #000073, 0 0 25px #000073, 0 0  25px #000073, 0 0 25px #000073;
            '>
                <p style='font-weight:bold; 
                color:#1E90FF  ;text-align:center ; font-size:33px; font-family:Verdana;'>Upload 3D MRI Scans of Patient below:   </p>
                <div style='margin-top:20px;'>
                    
            </div>
    
            """,
            unsafe_allow_html=True
        )
        st.write(" ")
        st.write(" ")

        t1ce_file = st.file_uploader("Choose a T1CE NIfTI file")
        t2_file = st.file_uploader("Choose a T2 NIfTI file")
        flair_file = st.file_uploader("Choose a FLAIR NIfTI file")
        
    

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


        if st.button("View Segmentation Maps"):
                st.write(" ")
                st.write(" ")
                st.write(" ")
                if t1ce_file is not None and t2_file is not None and flair_file is not None:
                    
                    if not os.path.exists("uploads"):
                        os.mkdir("uploads")
            
                    t1ce_path = os.path.join("uploads", t1ce_file.name)
                    t2_path = os.path.join("uploads", t2_file.name)
                    flair_path = os.path.join("uploads", flair_file.name)
                    flair_filename = os.path.basename(flair_path)
                    image_number = flair_filename[17:20]
                    parent_directory = r"D:\pfm\Python_files\archive1\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData_1"
                    mask_path = os.path.join(parent_directory, f"BraTS20_Training_{image_number}", f"BraTS20_Training_{image_number}_seg.nii")
            
                    with open(t1ce_path, "wb") as f:
                        f.write(t1ce_file.getbuffer())
                    with open(t2_path, "wb") as f:
                        f.write(t2_file.getbuffer())
                    with open(flair_path, "wb") as f:
                        f.write(flair_file.getbuffer())
            
                    t1ce_img = nib.load(t1ce_path).get_fdata()
                    t2_img = nib.load(t2_path).get_fdata()
                    flair_img = nib.load(flair_path).get_fdata()
            
                    mask_img = nib.load(mask_path)
                    mask_img = mask_img.get_fdata()
                    mask_img = mask_img.astype(np.uint8)
                    mask_img = mask_img[56:184, 56:184, 13:141]
            
                    cropped_img = preprocess(t1ce_img, t2_img, flair_img)
                    prediction = predict_function(cropped_img)
                    random_slice= np.random.randint(0,127)
                    fig = plot_slices(cropped_img, mask_img, prediction,60)
                    st.write(" ")
                    st.write(" ")

    with col3:
        st.write(" ")
        st.write(" ")
    
        st.markdown(
            """
            <div style='background-color:#000000; 
            padding:20px; border: 5px solid #000073;
            box-shadow: 0 0 20px #000073, 0 0 25px #000073, 0 0  25px #000073, 0 0 25px #000073;
            '>
                <p style='font-weight:bold; 
                color:#1E90FF  ; font-size:32px; font-family:Verdana; text-align:center;'>
                Output Segmentation Maps (A Random 2D Slice): </p>
                <div style='margin-top:20px;'>
                    
            </div>
    
            """,
            unsafe_allow_html=True
        )
        empty_space = st.empty()
        empty_space.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

        if fig is not None:
            st.pyplot(fig)
       
if __name__ == "__main__":
    main()