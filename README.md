# Automatic Brain Tumor Image Segmentation from Multi-Modal 3D MRI Scans Using U-Net Architecture (with Graphical User Interface).

     
     
![plot1y](https://user-images.githubusercontent.com/111432785/234045376-99493ee3-bc3a-41f7-8795-300778fff09c.png)
![plot3y](https://user-images.githubusercontent.com/111432785/234045394-e698011d-3185-4e64-9ce7-fa639704329e.png)



# Project Overview
Medical images, particularly from MRI or CT scans, are highly detailed but can be time-consuming to interpret. Traditional methods involve manual tracing of areas like tumors, which is painstaking, especially for complex shapes. This process, coupled with the need for precise verification, can affect radiologists' efficiency and patient care. Automated segmentation methods ease this burden, allowing radiologists to focus on verifying results and making accurate clinical decisions. This ultimately improves patient care by reducing diagnostic errors and enhancing treatment planning
The task involved developing a 3D U-Net model using Convolutional Neural Networks (CNNs) to perform accurate segmentation of brain tumors. Within the Brats2020 dataset, the data for 185 patients was organized into folders. These folders contained 3D MRI scans of multiple modalities in the NIfTI format.

Developed a 3D U-Net model using CNNs to accurately segment brain tumors from the Brats2020 dataset. The U-Net model achieved an average Dice coefficient of 0.844 indicating high accuracy in brain tumor segmentation and it is competitive with state-of-the-art methods. This model enhances diagnostic accuracy, aiding radiologists and neurologists. 

Subsequently, integrated the project into a full-stack application using Streamlit for both the frontend and backend, with an elegant UI using HTML and CSS, bridging frontend and backend development for seamless doctor interaction.
