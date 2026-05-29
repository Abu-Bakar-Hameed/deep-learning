# Ann.py
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =====================================================
# LOAD DATA
# =====================================================

file_path="C:/Users/24SW084/Desktop/ANN project/backend/credit_card_fraud_1000.csv"

data = pd.read_csv(file_path)

X = data.drop("Class", axis=1)
y = data["Class"]

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# BUILD MODEL
# =====================================================

model = tf.keras.Sequential([

    tf.keras.layers.Dense(
        32,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(8, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# CLASS WEIGHTS
# =====================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    i: class_weights[i]
    for i in range(len(class_weights))
}

# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save("fraud_ann_model.h5")

print("Model Training Complete ✅")

# =====================================================
# PREDICTIONS
# =====================================================

# Probability predictions

y_prob = model.predict(X_test)

# Convert probability -> binary

y_pred = (y_prob > 0.5).astype(int)

# =====================================================
# METRICS
# =====================================================

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)


