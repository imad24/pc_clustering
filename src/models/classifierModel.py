class classifierModel:
    
    def __init__(self, le_encoder=None, ohe_encoder=None, scaler=None, model=None, non_categorical=None, categorical=None ):
        self.le_encoder = le_encoder
        self.ohe_encoder = ohe_encoder
        self.scaler = scaler
        self.model = model
        self.non_categorical = non_categorical
        self.categorical = categorical


    def fit(self,X):
        self.model.fit(X)

    def predict(self,X):
        self.model.predict(X)
