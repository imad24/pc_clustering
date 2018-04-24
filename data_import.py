# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:36:05 2018

@author: amariller
"""




import pandas as pd
import math
import numpy as np

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import copy as cp

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

#import sompy

data_path = "..\\data\\raw\\"

#Product and promo files
df_product = pd.read_csv(data_path + "data_prod_bnd_ita.csv", sep = ";", encoding = 'iso8859_2')
# df_promo = pd.read_csv(data_path + "data_promo_bnd_ita2.csv", sep = ";", encoding = 'iso8859_2')


#History files for P2C1 and P2C4
df_histo_p2c1 = pd.read_csv(data_path + "data_histo_bnd_ita_p2c1.csv", sep = ";", encoding = 'iso8859_2', header = 0)
df_histo_p2c4 = pd.read_csv(data_path + "data_histo_bnd_ita_p2c4.csv", sep = ";", encoding = 'iso8859_2', header = 0)


#Client files
df_client = pd.read_csv(data_path + "data_client_bnd_ita.csv", sep = ";", encoding = 'iso8859_2', header = 0)


def get_list_client_lvl1(str_ClientPromo, df_AllClient):
    

    list_client = []
    
    for index, row in df_client.iterrows():
       
        if (row['Key_lvl1'] == "1" 
		or row['Key_lvl2'] == str_ClientPromo 
		or row['Key_lvl3'] == str_ClientPromo 
		or row['Key_lvl4'] == str_ClientPromo 
		or row['Key_lvl5'] == str_ClientPromo 
		or row['Key_lvl6'] == str_ClientPromo ):                     
            list_client.append(row['Key_lvl1'])
            
    return list_client



def get_list_client_lvl4(str_ClientPromo, df_AllClient):
    
    list_client = []
    
    for index, row in df_client.iterrows():
       
        if (row['Key_lvl1'] == str_ClientPromo or row['Key_lvl2'] == str_ClientPromo or row['Key_lvl3'] == str_ClientPromo or row['Key_lvl4'] == str_ClientPromo or row['Key_lvl5'] == str_ClientPromo or row['Key_lvl6'] == str_ClientPromo ):                     
           
            list_client.append(row['Key_lvl4'])
            
    return list(set(list_client))
    



#test_promo_cli = df_promo.at[6044,'Client']   #6044
test_promo_cli = '1'   
#test_promo_cli  = '68Q061H60CQ9'         #68Q061H60CQ9, promo 5586



list_lvl1_cli = get_list_client_lvl1(test_promo_cli, df_client)
list_lvl4_cli = get_list_client_lvl4(test_promo_cli, df_client)




df_HistoCli_p2c1 = df_histo_p2c1.loc[df_histo_p2c1['Client'].isin(list_lvl1_cli)]
    
list_UnqProd = list(set(df_HistoCli_p2c1['Product']))

df_HistPerProduct_p2c1 = pd.DataFrame(columns = df_HistoCli_p2c1.columns, index = list_UnqProd)


for str_prod in list_UnqProd:

    #df_temp = df_HistoCli[df_HistoCli['Product'] == list_UnqProd[1]]
    df_temp = df_HistoCli_p2c1[df_HistoCli_p2c1['Product'] == str_prod]
    
    df_test = df_temp.sum(axis = 0, skipna = True)
        
    df_HistPerProduct_p2c1.loc[str_prod] = df_test
    
    
df_HistPerProduct_p2c1.drop(['Product', 'Client'], axis = 1, inplace = True) 
    
df_HistPerProduct_p2c1 = df_HistPerProduct_p2c1[df_HistPerProduct_p2c1.sum(axis = 1) != 0]

df_HistPerProduct_p2c1 = df_HistPerProduct_p2c1[~np.isnan(df_HistPerProduct_p2c1.sum(axis = 1))]



df_HistoCli_p2c4 = df_histo_p2c4.loc[df_histo_p2c4['Client'].isin(list_lvl4_cli)]
    
list_UnqProd = list(set(df_HistoCli_p2c4['Product']))

df_HistPerProduct_p2c4 = pd.DataFrame(columns = df_HistoCli_p2c4.columns, index = list_UnqProd)


for str_prod in list_UnqProd:

    #df_temp = df_HistoCli[df_HistoCli['Product'] == list_UnqProd[1]]
    df_temp = df_HistoCli_p2c4[df_HistoCli_p2c4['Product'] == str_prod]
    
    df_test = df_temp.sum(axis = 0, skipna = True)
        
    df_HistPerProduct_p2c4.loc[str_prod] = df_test
    
    
df_HistPerProduct_p2c4.drop(['Product', 'Client'], axis = 1, inplace = True) 
    

df_HistPerProduct_p2c4 = df_HistPerProduct_p2c4[df_HistPerProduct_p2c4.sum(axis = 1) != 0]

df_HistPerProduct_p2c4 = df_HistPerProduct_p2c4[~np.isnan(df_HistPerProduct_p2c4.sum(axis = 1))]


list_it = list(range(len(df_HistPerProduct_p2c1.columns)))


tick_frequency = 4

for index, row in df_HistPerProduct_p2c1.iterrows():

    #print(index)
    #plt.plot(df_HistPerProduct_p2c1.ix[index])
    
    plt.plot(list(row))
    plt.title(str(index))

    plt.xticks(list_it[::tick_frequency], list(df_HistPerProduct_p2c1.columns)[::tick_frequency], rotation = 70)
    
    plt.show()



#df_HistPerProduct_p2c1.fillna(value = -1, inplace = True)
#df_HistPerProduct_p2c4.fillna(value = -1, inplace = True)
#
#testeq = df_HistPerProduct_p2c1 == df_HistPerProduct_p2c4



