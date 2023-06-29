import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle


def Classification_Model_to_find_Lost_and_Won_customer():
       quantity = st.sidebar.number_input('Enter quantity(max 150 tons and min 1 tons): ')
       thickness = st.sidebar.number_input('Enter thickness (Value between 0.1 to 7): ')
       width = st.sidebar.number_input('Enter width(Value between 2000  and  700 ): ')
       selling_price = st.sidebar.number_input('Enter selling price(Value between 1500  and  250 ): ')
       delivery_duration = st.sidebar.number_input('Enter Delivery_duration(Value between 1  and  200 days): ')
       item_type_lbenc = st.sidebar.selectbox('Search the Item type ',('IPL', 'Others', 'PL', 'S', 'SLAWR', 'W', 'WI'))
       country_lbenc = st.sidebar.selectbox('Select the country',('107',  '89',  '40',  '80',  '77',  '79',  '39',  '78',  '27',  '26',  '32', '28',  '25',  '84',  '30',  '38', '113'))
       application_lbenc = st.sidebar.selectbox('select the application',('58', '68', '59', '28', '40', '25', '15', '56', '39', '66', '69', '22',  '3', '10', '26', '27',  '4',  '5',  '2', '19', '29', '67', '20', '70', '42', '79', '65', '38', '41', '99'))
       product_ref_lbenc = st.sidebar.number_input('Enter Product ref: ')
       material_ref_lbenc = st.sidebar.number_input('Enter material ref: ')




       county = {'107':1,  '89':2,  '40':3,  '80':4,  '77':5,  '79':6,  '39':7,  '78':8,  '27':9, '26':10,  '32':11, '28':12,  '25':13,  '84':14,  '30':15,  '38':16, '113':17}

       item_type = {'W':6, 'S':4, 'Others':5, 'PL':2, 'WI':1, 'IPL':3, 'SLAWR':7}
       app = {'58':1, '68':2, '59':3, '28':4, '40':5, '25':6, '15':7, '56':8, '39':9, '66':10, '69':11, '22':12,  '3':13, '10':14, '26':15, '27':16,  '4':17,  '5':18,  '2':19, '19':20, '29':21, '67':22, '20':23, '70':24, '42':25, '79':26, '65':27, '38':28, '41':29, '99':30}

       item = item_type.get(item_type_lbenc)
       apps = app.get(application_lbenc)
       countrys = county.get(country_lbenc)
       logistic_regression = pickle.load(open(r'C:\Users\kisho\Downloads\Copperset_ML\logistic_regression.pkl','rb'))
       xgbc = pickle.load(open(r'C:\Users\kisho\Downloads\Copperset_ML\xgbc.pkl','rb'))
       clf = pickle.load(open(r'C:\Users\kisho\Downloads\Copperset_ML\clf.pkl','rb'))

       data = np.array([quantity, thickness, width, selling_price, int(delivery_duration), int(item), int(countrys), int(apps), int(product_ref_lbenc), int(material_ref_lbenc)])
       xtest = data.reshape(1, -1)

#st.write(quantity, thickness, width, selling_price, int(delivery_duration), int(item), int(countrys), int(apps), int(product_ref_lbenc), int(material_ref_lbenc))
#st.write(xtest)

       st.write('XGBC')
       st.write(xgbc.predict(xtest))
       st.write('Logistic_regression')
       st.write(logistic_regression.predict(xtest))
       st.write('Extra_tree_CLF')
       st.write(clf.predict(xtest))
       st.write("We have used the 3 different classification model(Logistics Regression, XG BOOST Classifier, Extra_tree_Classifier) for our study purpose ")

       st.write("XGBC and Extra_tree_CLF is better model with 90% of accuracy achieved compared to logistic regression model with 80% accuracy achieved for this data.")


def Regression_model_To_Predict_Selling_price():
       quantity = st.sidebar.number_input('Enter quantity(max 150 tons and min 1 tons): ')
       thickness = st.sidebar.number_input('Enter thickness (Value between 0.1 to 7): ')
       width = st.sidebar.number_input('Enter width(Value between 2000  and  700 ): ')
       delivery_duration = st.sidebar.number_input('Enter Delivery_duration(Value between 1  and  200 days): ')
       item_type_lbenc = st.sidebar.selectbox('Search the Item type ', ('IPL', 'Others', 'PL', 'S', 'SLAWR', 'W', 'WI'))
       country_lbenc = st.sidebar.selectbox('Select the country', (
       '107', '89', '40', '80', '77', '79', '39', '78', '27', '26', '32', '28', '25', '84', '30', '38', '113'))
       application_lbenc = st.sidebar.selectbox('select the application', (
       '58', '68', '59', '28', '40', '25', '15', '56', '39', '66', '69', '22', '3', '10', '26', '27', '4', '5', '2',
       '19', '29', '67', '20', '70', '42', '79', '65', '38', '41', '99'))
       product_ref_lbenc = st.sidebar.number_input('Enter Product ref: ')
       material_ref_lbenc = st.sidebar.number_input('Enter material ref: ')

       county = {'107': 1, '89': 2, '40': 3, '80': 4, '77': 5, '79': 6, '39': 7, '78': 8, '27': 9, '26': 10, '32': 11,
                 '28': 12, '25': 13, '84': 14, '30': 15, '38': 16, '113': 17}

       item_type = {'W': 6, 'S': 4, 'Others': 5, 'PL': 2, 'WI': 1, 'IPL': 3, 'SLAWR': 7}
       app = {'58': 1, '68': 2, '59': 3, '28': 4, '40': 5, '25': 6, '15': 7, '56': 8, '39': 9, '66': 10, '69': 11,
              '22': 12, '3': 13, '10': 14, '26': 15, '27': 16, '4': 17, '5': 18, '2': 19, '19': 20, '29': 21, '67': 22,
              '20': 23, '70': 24, '42': 25, '79': 26, '65': 27, '38': 28, '41': 29, '99': 30}

       item = item_type.get(item_type_lbenc)
       apps = app.get(application_lbenc)
       countrys = county.get(country_lbenc)
       Lregression = pickle.load(open(r'C:\Users\kisho\Downloads\Copperset_ML\reg_best_reg.pkl', 'rb'))

       data = np.array([quantity, thickness, width, int(delivery_duration), int(item), int(countrys), int(apps),
                        int(product_ref_lbenc), int(material_ref_lbenc)])
       xtest = data.reshape(1, -1)

       # st.write(quantity, thickness, width, selling_price, int(delivery_duration), int(item), int(countrys), int(apps), int(product_ref_lbenc), int(material_ref_lbenc))
       # st.write(xtest)

       st.write('random_forest_algo')
       st.write('predicting selling price')
       st.write(Lregression.predict(xtest))
       st.write("used randomforest algorithm for the prediction of the selling price with 90% of accuracy achieved.")

page_names_to_funcs = {
    "Classification_Model_to_find_Lost_and_Won_customer": Classification_Model_to_find_Lost_and_Won_customer,
    "Regression_model_To_Predict_Selling_price": Regression_model_To_Predict_Selling_price,

}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()




