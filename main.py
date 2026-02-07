# import streamlit as st
# import numpy as np
# import pickle
#
# # import the model
# pipe=pickle.load(open('pipe.pkl','rb'))
# df=pickle.load(open('df.pkl','rb'))
#
# st.title("Laptop Predictor")
#
# # brand
# company= st.selectbox('Brand',df['Company'].unique())
#
# # type of laptop
# type=st.selectbox('Type',df['TypeName'].unique())
#
# # Ram
# ram=st.selectbox('RAM(in GB)',[2,4,8,12,16,24,32,64])
#
# # Weight
#
# weight=st.number_input('Weight of the laptop')
#
# # touchscreen
#
# touchscreen=st.selectbox('Touchscreen',['No','Yes'])
#
# # IPS
# ips=st.selectbox('IPS',['No','Yes'])
#
# # screen size
# screen_size=st.number_input('Screen Size')
#
# # resolution
# resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2384x1440'])
#
# # CPU
#
# cpu=st.selectbox('CPU',df['Cpu brand'].unique())
#
# hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
#
# ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
#
# gpu=st.selectbox('GPU',df['Gpu brand'].unique())
#
# os=st.selectbox('OS',df['os'].unique())
#
#
# if st.button('Predict Price'):
#     # pass
#     ppi=None
#
#     if touchscreen=='Yes':
#         touchscreen=1
#     else:
#         touchscreen=0
#
#     if ips=='Yes':
#         ips=1
#     else:
#         ips=0
#
#     X_res=int(resolution.split('x')[0])
#     Y_res=int(resolution.split('x')[1])
#     ppi=((X_res**2)+(Y_res**2))**0.5/screen_size
#     ppi=()
#     query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
#
#     query=query.reshape(1,12)
#     st.title("The Predicted Price of this configuration is: "+str(int(np.exp(pipe.predict(query)[0]))))
#
#


import streamlit as st
import numpy as np
import pandas as pd
import pickle

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type_name = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the laptop')

# touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160',
     '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2384x1440']
)

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os_name = st.selectbox('OS', df['os'].unique())


# Predict button
if st.button('Predict Price'):

    # Convert Yes/No to 1/0
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size


    # Create DataFrame for pipeline
    input_df = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'Ips': ips,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os_name
    }])


    # Predict
    price = pipe.predict(input_df)[0]

    # Show result
    st.title("The Predicted Price of this configuration is: ₹ " + str(int(np.exp(price))))













