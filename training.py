import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib
import os

# Caminho para os dados
csv_path = 'data/alfabeto_libras.csv'

# Corrige o nome da primeira coluna
df = pd.read_csv(csv_path)
df.columns = ['letra'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]

X = df.drop('letra', axis=1).values
y = df['letra'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Cria diretório models/ se não existir
os.makedirs('models', exist_ok=True)

# Salva encoder
joblib.dump(label_encoder, 'models/label_encoder.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Salva modelo treinado
model.save('models/modelo_libras.h5')