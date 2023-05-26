# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:32:58 2023

@author: ANUJ
"""

import os
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
import streamlit as st
from utils import preprocess, predict_function, plot_slices
st.set_page_config(page_title="Brats2019_WebPage",page_icon=":brain:", layout="wide")

def main():

    fig = None
    st.markdown(
        """
        <div style='background-color:#000000; 
        padding:20px; border: 5px solid #000073;
        box-shadow: 0 0 25px #000073, 0 0 25px #000073, 0 0  25px #000073, 0 0 25px #000073;
        '>
            <p style='font-weight:bold; 
            color:#039aff  ; font-size:32px; font-family:Verdana; text-align:center'>
            You are currently working on the Brats2019 dataset.
            </p>
            <div style='margin-top:20px;'>
                
        </div>

        """,
        unsafe_allow_html=True
    )
    
    
    col1, col2, col3= st.columns([18,1.5,19])
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
                color:#039aff  ;text-align:center ; font-size:33px; font-family:Verdana;'>Upload 3D MRI Scans of Patient below:            
                </p>
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


        st.write(" ")
        st.write(" ")  
                
        if st.button("View Segmentation Maps"):
    
            if t1ce_file is not None and t2_file is not None and flair_file is not None:
                for file in [t1ce_file, t2_file, flair_file]:
                    if not file.name.endswith(".nii"):
                        st.error("Error: The selected file must be in NIfTI format .nii . Please select a file with the correct format and try again.")
                        return
                    
            
                t1ce_path = os.path.join("uploads", t1ce_file.name)
                t2_path = os.path.join("uploads", t2_file.name)
                flair_path = os.path.join("uploads", flair_file.name)
                flair_filename = os.path.basename(flair_path)
                
                glioma_grade = flair_filename[0:3]
                if glioma_grade == "HGG":   
                    parent_dir_of_mask = flair_filename[4:-10]
                    mask_path = "D:/brats2019/MICCAI_BraTS_2019_Data_Training/HGG/{}/HGG_{}_seg.nii".format(parent_dir_of_mask, parent_dir_of_mask)
                    mask_array = nib.load(mask_path).get_fdata()  
                    mask_img = mask_array.astype(np.uint8)
                else:
                    parent_dir_of_mask = flair_filename[4:-10]
                    mask_path = "D:/brats2019/MICCAI_BraTS_2019_Data_Training/LGG/{}/LGG_{}_seg.nii".format(parent_dir_of_mask, parent_dir_of_mask)
                    mask_array = nib.load(mask_path).get_fdata()  
                    mask_img = mask_array.astype(np.uint8)
                
                with open(t1ce_path, "wb") as f:
                    f.write(t1ce_file.getbuffer())
                with open(t2_path, "wb") as f:
                    f.write(t2_file.getbuffer())
                with open(flair_path, "wb") as f:
                    f.write(flair_file.getbuffer())
            
                t1ce_img = nib.load(t1ce_path).get_fdata()
                t2_img = nib.load(t2_path).get_fdata()
                flair_img = nib.load(flair_path).get_fdata()
            
                mask_img=mask_img[56:184,56:184,13:141]
            
                cropped_img = preprocess(t1ce_img, t2_img, flair_img)
                prediction=predict_function(cropped_img)
                fig = plot_slices(cropped_img, mask_img, prediction, 60)
                
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
                color:#039aff; font-size:30px; font-family:Verdana; text-align:center;'>
                Output Segmentation Maps (A Random 2D Slice): </p>
                <div style='margin-top:20px;'>
                    
            </div>
    
            """,
            unsafe_allow_html=True
        )
        empty_space = st.empty()
        empty_space.markdown("<div style='height:66.5px;'></div>", unsafe_allow_html=True)

        if fig is not None:
            st.pyplot(fig)
    
             
       
if __name__ == "__main__":

    main() 

       
