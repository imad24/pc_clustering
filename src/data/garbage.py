

def label_encoders(df):
    
    df = encode(df)
    le_dict = {}
    for index,col in df.iteritems():
        le = LabelEncoder()
        le_dict[index] = le.fit(col)
    return le_dict
    
    
def one_hot_encoders(label_encoders):
    ohe_dict ={}
    for key, value in label_encoders.items():
        ohe = OneHotEncoder()
        ohe_dict[key] = ohe.fit(value)
    return ohe_dict