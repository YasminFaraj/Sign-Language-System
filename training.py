import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Carrega os dados
df = pd.read_csv('alfabeto_libras.csv')
X = df.drop('letra', axis=1).values
y = df['letra'].values

# Codifica os rótulos (letras)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Salva o codificador para uso posterior
joblib.dump(label_encoder, 'label_encoder.pkl')

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)

# Cria o modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treina o modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Salva o modelo
model.save('modelo_libras.h5')
