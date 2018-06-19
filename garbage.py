from googletrans import Translator as Translator

col = 2
translator = Translator()
s = list(product_cluster.iloc[:,col:col+1].values.ravel())
t = translator.translate(s, dest='en')
translated =[]
for u in t:
    translated.append(u.text)
    
    
Key_lvl6 = np.array(translated).copy()

table = np.vstack((description,Key_lvl3,Key_lvl4,Key_lvl5,Key_lvl6)).T
print(table.shape)
translations = pd.DataFrame(table,columns = ["Description",[])

prp.save_file(translations,"translations")



[0, 0, 3]