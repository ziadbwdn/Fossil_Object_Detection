# initiate app with streamlit

import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import shutil

# Coba Import YOLO

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
st.set_page_config(page_title="Fossils Image Recognition")

# Periksa apakah library YOLO tersedia
def check_library():
    if not YOLO_AVAILABLE:
        st.error("Ultralytics are not installed. Install it first with this following command:")
        st.code("pip install ultralytics")
        return False
    return True

st.markdown("""
<div style="background-color:#0984e3; padding: 20px; text-align: center;">
<h1 style="color: white;"> Image Recognition Program </h1>
<h5 style="color: white;">Detection on Fossil Images</h5>
</div>
""", unsafe_allow_html=True)

# Pastikan library sudah terpasang sebelum melanjutkan
if check_library():
    # upload gambar
     uploaded_file = st.file_uploader("upload Invertebrate Fossil Images", type=['jpg', 'jpeg', 'png'])

     if uploaded_file:
        # Simpan sementara
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "images.jpg")
        image = Image.open(uploaded_file)
        
        #Ubah Ukuran Gambar
        image = image.resize((300,300))
        image.save(temp_file)

        #Tampilkan gamabar
        st.markdown("<div style='text-align: center;'>",unsafe_allow_html=True)
        st.image(image, caption="Uploaded Images")
        st.markdown("</div>", unsafe_allow_html=True)
        
        #Deteksi Gambar
        if st.button("Detecting Images"):
          with st.spinner("On Progress"):
              try:
                  model = YOLO('best.pt')
                  res = model(temp_file)
                  
                  #Ambil Hasil Prediksi
                  obj_name = res[0].names
                  pred_value = res[0].probs.data.numpy().tolist()
                  objek_terdeteksi = obj_name[np.argmax(pred_value)]
                  
                  #buat grafik
                  graph = go.Figure([go.Bar(x=list(obj_name.values()), y=pred_value)])
                  graph.update_layout(title='Degree of Accuracy Prediction', xaxis_title='Fossils',
                  yaxis_title='Accuracy')
                  
                  #Tampilkan hasil
                  st.write(f"Fossils Detected:{objek_terdeteksi}")
                  st.plotly_chart(graph)
                
              except Exception as e :
                  st.error("Images are unable to detect")
                  st.error(f"Error:{e}")
                  
              #Hapus file sementara
              shutil.rmtree(temp_dir,ignore_errors=True)

st.markdown(
"<div style='text-align: center;' class='footer'>Application Program on Fossil Detection @2025</div>",
unsafe_allow_html=True
)