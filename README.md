# Automatic Brain Tumor Image Segmentation from Multi-Modal 3D MRI Scans Using U-Net Architecture (with Graphical User Interface).

     
     
![plot1y](https://user-images.githubusercontent.com/111432785/234045376-99493ee3-bc3a-41f7-8795-300778fff09c.png)
![plot3y](https://user-images.githubusercontent.com/111432785/234045394-e698011d-3185-4e64-9ce7-fa639704329e.png)

##

# Description
Medical images, especially those from MRI or CT scans, can be intricate and densely detailed. Distinguishing between healthy and abnormal tissue, like tumors, often requires meticulous examination, which can be time-consuming. Traditional segmentation methods often involve manual tracing or outlining of regions of interest, such as tumors. This process is painstaking and can take a considerable amount of time, particularly for complex or irregularly shaped tumors. Precision in medical image segmentation is crucial but can be time-consuming for radiologists due to factors such as complex images, manual tracing, large datasets, variability, and the need for meticulous verification. These time constraints can impact radiologists' overall efficiency and patient care. Automated segmentation methods can alleviate this burden, enabling radiologists to give more importance and attention to the processes of confirming the accuracy of the segmentation results and making clinical decisions based on those results. This, in turn, enhances the quality of patient care by reducing the risk of diagnostic errors and supporting more effective treatment planning.

The task involved developing a 3D U-Net model using Convolutional Neural Networks (CNNs) to perform accurate segmentation of brain tumors. Within the Brats2020 dataset, the data for 185 patients was organized into folders. These folders contained 3D MRI scans of multiple modalities in the NIfTI format.

Developed a 3D U-Net model using CNNs to accurately segment brain tumors from the Brats2020 dataset. The U-Net model achieved an average Dice coefficient of 0.844 indicating high accuracy in brain tumor segmentation and it is competitive with state-of-the-art methods. This model enhances diagnostic accuracy, aiding radiologists and neurologists. 

Subsequently, integrated the project into a full-stack application using Streamlit for both the frontend and backend, with an elegant UI using HTML and CSS, bridging frontend and backend development for seamless doctor interaction.
