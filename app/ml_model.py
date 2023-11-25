from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np

def train_and_evaluate(classifier_name, param1, param2, param3):
    from sklearn.datasets import load_iris
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Seleciona o classificador
    if classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=int(param1))
    elif classifier_name == 'SVM':
        classifier = SVC(C=param1, kernel='rbf', gamma=param2)
    elif classifier_name == 'MLP':
        classifier = MLPClassifier(hidden_layer_sizes=(int(param1),), max_iter=int(param2))
    elif classifier_name == 'DT':
        classifier = DecisionTreeClassifier(max_depth=int(param1), min_samples_split=int(param2))
    elif classifier_name == 'RF':
        classifier = RandomForestClassifier(n_estimators=int(param1), max_depth=int(param2), min_samples_split=int(param3))

    # Treina o modelo
    classifier.fit(X_train, y_train)

    # Fazer previsões
    y_pred = classifier.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    # Plotar a matriz de confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf8')
    plt.close()

    # Resultados a serem exibidos no template
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'img_data': img_data,
        'classifier_name': classifier_name,
        'params': {'param1': param1, 'param2': param2, 'param3': param3}
    }

    return results


