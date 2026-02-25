SolarGuard – Solar Panel Defect Detection

This project uses Deep Learning (CNN) to classify solar panel defects.

Files:

train_cnn.py  
Trains CNN model using image dataset.

app.py  
Streamlit application to upload image and predict defect type.

Pipeline:

Images → CNN Model → Streamlit

How to Run:

python train_cnn.py  
python -m streamlit run app.py  

Output:

Predicted solar panel defect category.
