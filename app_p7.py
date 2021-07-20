
# coding: utf-8

# In[ ]:


# This is a sample Python script.
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

#loaing model and data
df_test=pd.read_csv('test.csv')#original data without target
X_test=pd.read_csv('df_X_test.csv')#data transformed with predict probability and target
# read pickle files


#explainer_ = open("explainer.pkl","rb")
#explainer = pickle.load(explainer_)

shap_values_ = open("shap_values.pkl","rb")
shap_values = pickle.load(shap_values_)

features_selected_in = open("features_selected.pkl","rb")
features_selected = pickle.load(features_selected_in)

xgb_model_in = open("xgb_model.pkl","rb")
xgb_model = pickle.load(xgb_model_in)

X_shap=X_test[features_selected]
explainer = shap.TreeExplainer(xgb_model)
#shap_values = explainer(X_shap)

def get_sk_id_list():
    # Getting the values of SK_IDS from the content
    SK_IDS = df_test['SK_ID_CURR']
    return SK_IDS
sk_id_list = get_sk_id_list()

#select Client ID
st.sidebar.header("Select ID Client")
sk_id_curr = st.sidebar.selectbox('Select SK_ID from list:', sk_id_list, 0)
st.write('Client ID: ', sk_id_curr)

#Display caracteristic of Client
donnee=df_test[df_test['SK_ID_CURR']==sk_id_curr].drop(['Unnamed: 0'], axis=1)

st.subheader('Les Information du client')
st.write(donnee)

st.subheader('Decision')
prob=X_test[X_test['SK_ID_CURR']==sk_id_curr][['Target_prob','Target_pred']]
if prob['Target_pred']._values == 0:
    st.write('### Credit accepté')
else:
    st.write('### Credit rejeté')

st.write(prob)

submit = st.button('Get explain')
# explain model prediction results

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

if submit:
    j=prob.index.values
    j=int(j)
    #st.write(j)
    #shap.waterfall_plot(explainer.expected_value, shap_values[j], X_shap.iloc[j])
    #shap.plots.waterfall(shap_values[j])
    #explainer force_plot
    shap.initjs()

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    st.subheader('Model Prediction Interpretation Plot')
    st_shap(shap.force_plot(explainer.expected_value, shap_values.values[j], X_shap.iloc[j]))

    st.subheader('Summary Plot 1')


    nb_features=10

    #
    shap.plots.waterfall(shap_values[j])
    plt.gcf().set_size_inches(14, nb_features / 2)
    # Plot the graph on the dashboard
    st.pyplot(plt.gcf())

    st.subheader('Summary Plot 1')
    shap.summary_plot(shap_values, X_shap, plot_type="bar")
    plt.gcf().set_size_inches(14, nb_features / 2)
    # Plot the graph on the dashboard
    st.pyplot(plt.gcf())

