import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

st.title('การทดสอบการเขียนเว็บด้วย Streamlit')
col1, col2 = st.columns(2)
#col1.write("This is column 1")
#col2.write("This is column 2")
with col1:
   #st.image("./pic/kairung02.jpg")
   st.header("ไก้รุ่ง เฮงพระพรหม")
with col2:
   #st.header("Versicolor")
   st.header("ไก้รุ่ง เฮงพระพรหม")
   st.text('สาขาวิชาวิทยาการข้อมูล')
   st.write('คณะวิทยาศาสตร์และเทคโนโลยี')
   st.markdown('มหาวิทยาลัยราชภัฏนครปฐม')

html_1 = """
<div style="background-color:#FFBF00;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

colx1, colx2,colx3 = st.columns(3)
#col1.write("This is column 1")
#col2.write("This is column 2")
with colx1:
   st.image("./pic/iris.jpg")
with colx2:
   st.image("./pic/iris02.jfif")
with colx3:
   st.image("./pic/iris1.jpg")

#import pandas as pd
dt=pd.read_csv('./data/iris.csv')
st.write(dt.head(10))

dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("show chart"):
   st.bar_chart(dx2)
   st.button("Don't show chart")
else:
   st.button("Don't show chart")

html_2 = """
<div style="background-color:#FFBF00;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายคลาสดอกไม้</h5></center>
</div>
"""
st.markdown(html_2, unsafe_allow_html=True)
st.markdown("")

ptlen = st.slider("กรุณาเลือกข้อมูล petal.length",0,10)
ptwd = st.slider("กรุณาเลือกข้อมูล petal.width",0,10)

splen = st.number_input("กรุณาเลือกข้อมูล sepal.length")
spwd = st.number_input("กรุณาเลือกข้อมูล sepal.width")


if st.button("ทำนายผล"):
   #dt = pd.read_csv("./data/iris.csv") 
   X = dt.drop('variety', axis=1)
   y = dt.variety   

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)

   x_input = np.array([[ptlen, ptwd, splen, spwd]])

   st.write(Knn_model.predict(x_input))
   out=Knn_model.predict(x_input)
   if out[0]=="Setosa":
      st.image("./pic/iris.jpg")
   elif out[0]=="Versicolor":
      st.image("./pic/iris2.jpg")
   else:
      st.image("./pic/iris3.jpg")  
   st.button("ไม่ทำนาย")
else:
   st.button("ไม่ทำนาย")