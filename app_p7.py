import pickle
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import plotly.express as px
from statistics import *
import math


st.markdown("<h1 style='text-align: center; color: #ff634d;'><strong><u>Home Credit Default Risk</u></strong></h1>",
            unsafe_allow_html=True)

title_image = Image.open("pret.png")
st.sidebar.image(title_image)

#loaing model and data
df_test=pd.read_csv('test_var.csv')#original data without target
X_test=pd.read_csv('df_X_test.csv')#data transformed with predict probability and target
# read pickle files
#df_test = df_test.drop(['AMT_TOTAL_RECEIVABLE'], axis=1)

#shap values
shap_values_ = open("shap_values.pkl","rb")
shap_value = pickle.load(shap_values_)
#features
features_selected_in = open("features_selected.pkl","rb")
features_selected = pickle.load(features_selected_in)
#load lodel
xgb_model_in = open("Best_model.pkl","rb")
xgb_model = pickle.load(xgb_model_in)
#explaner shap
X_shap=X_test[features_selected]
explainer = shap.TreeExplainer(xgb_model)
#shap_value = explainer(X_shap)


#shap_values = explainer(X_shap)
@st.cache()
def get_sk_id_list():
    # Getting the values of SK_IDS from the content
    SK_IDS = df_test['SK_ID_CURR']

    return SK_IDS
sk_id_list = get_sk_id_list()


#select Client ID
#st.sidebar.header("ID Client")
st.sidebar.markdown("<h1 style='text-align: center; color: black;'><strong><u>ID Client</u></strong></h1>",
                    unsafe_allow_html=True)
sk_id_curr = st.sidebar.selectbox('Choisir Dossier Client:', sk_id_list, 0)
#seuil
st.sidebar.header("Paramètre")
threshold = st.sidebar.slider(
        label='Threshold:',
        min_value=0.,
        value=0.5,
        max_value=1.)

#Display caracteristic of Client
#donnee=df_test[df_test['SK_ID_CURR']==sk_id_curr].drop(['Unnamed: 0'], axis=1)
donnee=df_test[df_test['SK_ID_CURR']==sk_id_curr].drop(['Unnamed: 0'], axis=1)
ix=df_test[df_test['SK_ID_CURR']==sk_id_curr].index

st.subheader('Les informations du client')
st.write(donnee)


st.subheader('Decision')
#prob=X_test[X_test['SK_ID_CURR']==sk_id_curr][['Target_prob','Target_pred']]
prob=X_test['Target_prob'].loc[ix]

prb=int(prob.values*100)
predicted_class = np.where(prb/100 > threshold, 1, 0)
st.markdown(f"**Default Risk:** {prb}**%**")

if predicted_class == 0:
    html_temp = """
        <div style="background-color:green">
        <h3 style="color:white;text-align:center;">Crédit Accepté </h3>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    #st.write('### Credit accepté')
else:
    #st.write('### Credit Rejeté')
    html_temp2 = """
            <div style="background-color:red">
            <h3 style="color:white;text-align:center;">Crédit Rejeté </h3>
            </div>
            """
    st.markdown(html_temp2, unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: black;'><strong><u>Interprétabilité </u></strong></h2>",
            unsafe_allow_html=True)

submit=st.button("Explication")

# explain model prediction results
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

if submit:
    j=prob.index.values
    j=int(j)
    shap.initjs()

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    st.subheader('Interprétation de la prédiction du modèle :Graphique 1')
    st_shap(shap.force_plot(explainer.expected_value, shap_value.values[j], X_shap.iloc[j]))

    st.subheader('Interprétation de la prédiction du modèle :Graphique 2 ')


    nb_features=20
    shap.plots.waterfall(shap_value[j])
    plt.gcf().set_size_inches(16, nb_features / 2)
    # Plot the graph on the dashboard
    st.pyplot(plt.gcf())

st.markdown("<h2 style='text-align: center; color: black;'><strong><u>Analyse</u></strong></h2>",
            unsafe_allow_html=True)
#Analyse
var=['CNT_CHILDREN', 'AMT_CREDIT_x', 'AMT_ANNUITY',
       'AMT_GOODS_PRICE','DAYS_REGISTRATION', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'Diff_day','Diff_pay', 'AMT_CREDIT_y', 'AMT_DOWN_PAYMENT',
       'NAME_CONTRACT_STATUS_Approved', 'NAME_CONTRACT_STATUS_Canceled',
       'NAME_CONTRACT_STATUS_Refused', 'REJECT_REASON', 'Status_active',
       'CNT_INSTALMENT_FUTURE', 'SK_DPD_DEF', 'AMT_DRAWINGS', 'MONTHS_BALANCE',
       'AMT_TOTAL_RECEIVABLE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE',
       'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'AMT_CREDIT_SUM',
       'Nombre_active', 'Comsumer_type_crd', 'Age', 'CREDIT_INCOME',
       'ANNUITY_INCOME', 'CREDIT_LENGTH', 'YEAR_EMPLOYED_PERCENT',
       'CREDIT_TO_GOODS_RATIO', 'PAYMENT_RATE', 'ANNUITY_INCOME_PERS']

variable=st.selectbox('Choisir la variable', var)
fig2 = px.histogram(df_test, x=variable)

x_val=int(donnee[variable].values)
fig2.add_vline(x=x_val, line_width=3, line_dash="dash", line_color="green",
               annotation_text="ID_Client: "+str(x_val),
               annotation_position="top right",)
st.plotly_chart(fig2)

st.markdown("<h2 style='text-align: center; color: black;'><strong><u>Comparaison</u></strong></h2>",
            unsafe_allow_html=True)

pred_df=df_test.copy()
pred_df['pred']=predicted_class
post_df=pred_df[pred_df.pred==1]
neg_df=pred_df[pred_df.pred==0]
mean_va_pos=post_df[variable].mean()
mean_va_neg=neg_df[variable].mean()
valeur=x_val
datatable = pd.DataFrame([valeur,mean_va_neg,mean_va_pos],
                   columns=['Client', 'Moyenne_Défaut', 'Moyenne_Non_Défaut'])
st.table(datatable)# will display the table
