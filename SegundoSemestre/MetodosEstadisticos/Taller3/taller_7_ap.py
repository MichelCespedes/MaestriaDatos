import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use("ggplot")
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt

# Se comparará el desempeño de DBSCAN vs. KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture as GMM
from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#from sklearn.datasets import load_iris
from sklearn import metrics 


from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics 

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus


from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import datasets
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

from sklearn import linear_model


def k_means(d_x,d_y,d_z):

    X=np.column_stack((d_x,d_y,d_z))
    kmeans=KMeans(n_clusters=3)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    plt.title("Cluster con n=3")
    plt.scatter(X[:,0], X[:,1], c=labels, s=7, cmap='viridis')
    plt.scatter(centroids[:, 0],centroids[:, 1], marker = "*", s=150, linewidths = 2, zorder = 10)
    plt.show()

    costo = []
    for i in range(1,8):
        kmeans = KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
        kmeans.fit(X)
        costo.append(kmeans.inertia_)
        print("Cluster", i, "Función de costo", kmeans.inertia_)

    plt.plot(range(1,8),costo)
    plt.title("Gráfica de codo")
    plt.xlabel("Número de clusters")
    plt.ylabel("Funcion de costo") 
    plt.show()

    plt.title("Cluster con etiqueta original")
    plt.scatter(X[:,0], X[:,1], c=X[:,2], s=7, cmap='viridis')


    for i in range(2,8): 
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(X)
        n_c=str(i)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        plt.title("Cluster con n="+n_c)
        plt.scatter(X[:,0], X[:,1], c=labels, s=7, cmap='viridis')
        plt.scatter(centroids[:, 0],centroids[:, 1], marker = "*", s=150, linewidths = 2, zorder = 10)

        plt.show()
        
        
def dbscan_(d_x,d_y,d_z):
    X=np.column_stack((d_x,d_y,d_z))

    data = X
    # Estandarizar los datos a la media y la desviación estándar
    for x in range(2):
        m = data[:,x].mean()
        s = data[:,x].std()
        for y in range(len(data)):
            data[y,x] = (data[y,x] - m)/s

    # DBSCAN desde Scikit-Learn
    dbscan = DBSCAN(eps=0.2, min_samples = 2)
    clusters = dbscan.fit_predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=clusters, s=7, linewidths = 2, zorder = 10)
    
    
     
def m_gauss(d_x,d_y,d_z):
#     %matplotlib inline 
    X=np.column_stack((d_x,d_y))
    X, y_true = X,d_z
    plt.scatter(X[:, 0], X[:, 1], s=7)
    plt.show()
    the_gmm = GMM(n_components=4).fit(X)
    labels = the_gmm.predict(X)
    print("Clasificación no-supervisada mediante mezcla de gaussianas")
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=7, cmap='viridis');
    plt.show()
    print("Etiqueta original, de la generación de los datos aleatorios")
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=7, cmap='viridis');
    plt.show()
    
    
    
def k_nn(datax,datay,class_n):
    X_train, X_test, y_train, y_test = train_test_split(datax,
                                                        datay, random_state=0)
    print("Tamaño de X_train: {}\nTamaño de y_train: {}".format(X_train.shape, y_train.shape))
    print("Tamaño de X_test: {}\nTamaño de y_test: {}".format(X_test.shape, y_test.shape))

    knn = KNeighborsClassifier(n_neighbors=1);
    knn.fit(X_train, y_train)


    y_pred = knn.predict(X_test)
    # comparación de las predicciones con las etiquetas originales
    pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicción', index=X_test.index)], 
              ignore_index=False, axis=1)
    print("Score sobre el conjunto de test: {:.3f}".format(knn.score(X_train, y_train)))

    np.set_printoptions(precision=2)
    cv_scores = cross_val_score(knn, X_test, y_test, cv=5)
    print (cv_scores);

    print('Promedio')
    print(np.average(cv_scores))

    # Escribe en pantalla la métrica "accuracy" para prueba (datos no usados en el entrenamiento)
    print("Exactitud (accuracy) en prueba (testing):",metrics.accuracy_score(y_test, y_pred))

    class_names = class_n

    disp = plot_confusion_matrix(knn, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title("Matriz de confusión sin normalización")

    disp = plot_confusion_matrix(knn, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title("Matriz de confusión normalizada")

    # print(disp.confusion_matrix)

    plt.show()
    
    
def tree_desc(datax,datay,class_n,class_n_2):
    for i in range(len(class_n_2)):
        class_n_2[i]=str(class_n_2[i])
    feature_cols =class_n
    X = datax # Características
    y = datay # Variable objetivo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
    print("Tamaño de X_train: {}\nTamaño de y_train: {}".format(X_train.shape, y_train.shape))
    print("Tamaño de X_test: {}\nTamaño de y_test: {}".format(X_test.shape, y_test.shape))
    dtree = DecisionTreeClassifier()

    # Entrenamiento
    dtree = dtree.fit(X_train,y_train)

    #Predicción para evaluación
    y_pred = dtree.predict(X_test)
    print(classification_report(y_test, y_pred,target_names=class_n_2))
#     cv_scores = cross_val_score(dtree, X_train, y_train, cv=5)

#     print (cv_scores);

#     print(np.average(cv_scores))

#     print("Exactitud (accuracy):",metrics.accuracy_score(y_test, y_pred))

#     print(X_test.iloc[210])
    # data = {'Embarazos':[2],'Insulina':[120.000],'Indice Masa Corporal':[20],'Edad':[32.000],'Glucosa':[190],'Presion Arterial':[100],'Función de Pedigree':[0.75]}
    # query=pd.DataFrame(data)

    # nueva_clasificacion=dtree.predict(query)

    # print("Clase: ",nueva_clasificacion)
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,  
                   filled=True, rounded=True,
                   special_characters=True, feature_names = list(datax.columns),class_names=class_n_2)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    return graph



def  svm_f(datax,datay,avera):


    X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.3,random_state=109) 

    print("Tamaño de X_train: {}\nTamaño de y_train: {}".format(X_train.shape, y_train.shape))
    print("Tamaño de X_test: {}\nTamaño de y_test: {}".format(X_test.shape, y_test.shape))

    #Crear un nuevo clasificador SVM 
    svm_class = svm.SVC(kernel='linear') # Linear Kernel

    #Se entrena ("ajusta") el modelo, usando los patrones de entrenamiento
    svm_class.fit(X_train, y_train)

    #Se obtiene el conjunt de salidas obtenidas con el modelo ajustado, según el dataset de prueba
    y_pred = svm_class.predict(X_test)
    print(classification_report(y_test, y_pred,target_names=avera))

    
    
def random_forest(datax,datay):
    from numpy import mean
    from numpy import std
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    # define dataset
    X =datax
    y =datay
    # define the model
#     X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.2,random_state=100) 
    model = RandomForestClassifier()
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred,target_names=class_n))
#     evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

def N_B(datax,datay,class_n):
    algoritmo = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.2,random_state=100) 
    class_names=class_n
    algoritmo.fit(X_train, y_train)
    y_pred = algoritmo.predict(X_test)
    disp = plot_confusion_matrix(algoritmo, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title("Matriz de confusión sin normalización")
    plt.xticks(rotation=90)
    disp = plot_confusion_matrix(algoritmo, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title("Matriz de confusión normalizada")
    plt.xticks(rotation=90)
    print(classification_report(y_test, y_pred,target_names=class_names))
    
def lineal_m(datax,datay):
    X_train, X_test, y_train, y_test = train_test_split(datax,datay, test_size=0.3, random_state=1) 
    model = linear_model.LogisticRegression()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    