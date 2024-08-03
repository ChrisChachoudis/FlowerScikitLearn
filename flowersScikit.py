import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Φόρτωση του dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Διαχωρισμός δεδομένων σε εκπαίδευση και δοκιμή
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Εκπαίδευση του μοντέλου λογιστικής παλινδρόμησης
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Πρόβλεψη στα δεδομένα δοκιμής
y_pred = model.predict(X_test)

# Αξιολόγηση του μοντέλου
accuracy = accuracy_score(y_test, y_pred)
print(f"Ακρίβεια του μοντέλου: {accuracy:.2f}")

# PCA για μείωση διαστάσεων σε 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Δημιουργία διαγράμματος
plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()