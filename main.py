from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Veri setini yükle
data = load_breast_cancer()

# DataFrame oluştur
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Özellikler ve hedef değişkeni ayır
X = df.drop('target', axis=1)
y = df['target']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN modeli oluştur
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Tahmin yap
y_pred = knn.predict(X_test)

# Sonuçları değerlendirme
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))