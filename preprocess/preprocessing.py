from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Create simple preprocessing class that use sklearn preprocessing MinMaxScaler and Normalizer
class Preprocessing:
    def __init__(self, data) -> None:
        self.data = data
        self.scaler = None
        self.normalizer = None
        self.label_encoder = None
        self.one_hot_encoder = None

    def scale(self, scaler_type):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            print('Invalid scaler type')
            return

        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        
        return self.data

    def normalize(self, normalizer_type):
        if normalizer_type == 'l1':
            self.normalizer = Normalizer(norm='l1')
        elif normalizer_type == 'l2':
            self.normalizer = Normalizer(norm='l2')
        else:
            print('Invalid normalizer type')
            return

        self.normalizer.fit(self.data)
        self.data = self.normalizer.transform(self.data)

    def encode_label(self, label_type):
        if label_type == 'onehot':
            self.one_hot_encoder = OneHotEncoder(sparse=False)
        elif label_type == 'label':
            self.label_encoder = LabelEncoder()
        else:
            print('Invalid label encoder type')
            return

        self.label_encoder.fit(self.data)
        self.data = self.label_encoder.transform(self.data)

    def decode_label(self, label_type):
        if label_type == 'onehot':
            self.one_hot_encoder = OneHotEncoder(sparse=False)
        elif label_type == 'label':
            self.label_encoder = LabelEncoder()
        else:
            print('Invalid label encoder type')
            return

        self.one_hot_encoder.fit(self.data)
