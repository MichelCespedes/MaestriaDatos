import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import prince

import scipy.stats as ss
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale

###############################################################
################ DESCRIPCION ###############################
###############################################################

def base(df):
    result = pd.DataFrame()
    
    columnas = df.columns.to_list()
    tipo = []
    unico = []
    nulls = []
    for col in df.columns.to_list():
        tipo.append(df[col].dtype)
        unico.append(len(df[col].unique()))
        nulls.append(len(df[df[col].isnull()]))
        
    result['Columnas'] = df.columns.to_list()
    result['Tipo'] = tipo
    result['Único'] = unico
    result['Nulos'] = nulls
    
    return result
    



###############################################################
################ CORRELACIONES ###############################
###############################################################

def correlacion_spearman(datos,anotacion,ancho=20,alto=15):

    cuanti=datos.select_dtypes(np.number)

    correlacion_global=datos.corr(method='spearman')


    rho,p_value = stats.spearmanr(cuanti)
    rho = list(p_value)
    for i in range(len(p_value)):
        p_value[i]= list(p_value[i])
    p_value = pd.DataFrame(p_value)
    Significancia = p_value < 0.05


    mask =np.triu(correlacion_global, k=1)

    sns.set(font_scale=1.7)
    fig, scatter = plt.subplots(figsize = (ancho,alto))
    sns.heatmap(data=correlacion_global.round(decimals=2), 
                xticklabels=correlacion_global.columns,
                yticklabels=correlacion_global.columns,
                cmap='RdBu_r',
                annot=anotacion,
                linewidth=0.5,
                mask=mask)
    

###############################################################
################ PCA ###############################
###############################################################

def matriz_componentes(df,corr):
    cuanti=df.select_dtypes(np.number)
    nombres = cuanti.columns
    
    if corr:
        scaler=StandardScaler()
        scaler.fit(cuanti)
        X_scaled=scaler.transform(cuanti)
    else:
        scaler=StandardScaler()
        scaler.fit(cuanti)
        X_scaled=cuanti 
    
    pca=PCA(n_components=len(nombres)) 
    pca.fit(X_scaled)
    cuanti_pca=pca.transform(X_scaled)
    
    if corr:
        pca_pipe = make_pipeline(StandardScaler(), PCA())
    else:
        pca_pipe = make_pipeline(PCA())
    
    pca_pipe.fit(cuanti)
    modelo_pca = pca_pipe.named_steps['pca']
    idx =[]
    for i in range(len(nombres)):
        lb = 'PC'+str(i+1)
        idx.append(lb)
    matriz_componentes = pd.DataFrame(
        data    = modelo_pca.components_,
        columns = cuanti.columns,
        index   = idx
    )
    
    return matriz_componentes,pca.components_

def graf_componentes(df,corr,ancho=5,largo=4):
    cuanti=df.select_dtypes(np.number)
    nombres = cuanti.columns
    
    fig, scatter = plt.subplots(figsize = (ancho,largo))
    
    _,c = matriz_componentes(df,corr)
    c=c.round(2)
    
    ACP=[]
    for i in range(0, len(c)):
      x=[" ACP", str(i+1)]
      a="".join(x)
      ACP.append(a)
    sns.set(font_scale=2)
    sns.heatmap(c,xticklabels=cuanti.columns, yticklabels=ACP, annot=True, cmap='coolwarm')
    
def biplot(df,pcax,pcay,corr=True,mx=0,my=0,ancho=15,largo=7):
    
    cuanti=df.select_dtypes(np.number)
    nombres = cuanti.columns
    
    fig, scatter = plt.subplots(figsize = (ancho,largo))
        
    nombres = cuanti.columns
    eti = []
    
    for i in range(len(nombres)):
        e  = nombres[i].split()
        eti.append(e[0])

        
    labels= eti
    
    scaler=StandardScaler()
    scaler.fit(cuanti)
    
    if corr:
        X_scaled=scaler.transform(cuanti)
    else:
        X_scaled=cuanti
    
    pca=PCA(n_components=len(nombres)) 
    pca.fit(X_scaled)
    cuanti_pca=pca.transform(X_scaled)

    if corr:
        pca_pipe = make_pipeline(StandardScaler(), PCA())
    else:
        pca_pipe = make_pipeline(PCA())
    
    
    pca1=pcax-1
    pca2=pcay-1
    xs = cuanti_pca[:,pca1]
    ys = cuanti_pca[:,pca2]
    n=pca.components_.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley)
    
    for i in range(len(nombres)):
        plt.arrow(0, 0, pca.components_[pca1,i], pca.components_[pca2,i],color='r',alpha=0.5) 
        if labels is None:
            plt.text(pca.components_[pca1,i]* 1.15, pca.components_[pca2,i] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(pca.components_[pca1,i]* 1.15, pca.components_[pca2,i] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1-mx,1+mx)
    plt.ylim(-1-my,1+my)
    
    sns.set_theme(style='white')
    sns.set(font_scale=1.5)
    sns.set_style('white')
    
    plt.xlabel("PC{}".format(pcax),size=20)
    plt.ylabel("PC{}".format(pcay),fontsize=20)
    plt.grid(False)
    

def var_acum(df,corr=True,ancho=20,largo=10):
    cuanti=df.select_dtypes(np.number)
    nombres = cuanti.columns    
    
    pca=PCA(n_components=len(cuanti)) 
    
    if corr:
        pca_pipe = make_pipeline(StandardScaler(), PCA())
    else:
        pca_pipe = make_pipeline(PCA())
        
    pca_pipe.fit(cuanti)
    modelo_pca = pca_pipe.named_steps['pca']

    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    print('------------------------------------------')
    print('Porcentaje de varianza explicada acumulada')
    print('------------------------------------------')
    print(prop_varianza_acum)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize = (ancho,largo))
    ax.plot(
        np.arange(len(cuanti.columns)) + 1,
        prop_varianza_acum,
        marker = 'o'
    )
    
    for x, y in zip(np.arange(len(cuanti.columns)) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
        
    ax.set_ylim(0, 1.19)
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza acumulada')
    
def df_pca(df,corr=True):

    
    cuanti=df.select_dtypes(np.number)
    nombres= cuanti.columns
    
    pca=PCA(n_components=len(cuanti)) 
    
    if corr:
        pca_pipe = make_pipeline(StandardScaler(), PCA())
    else:
        pca_pipe = make_pipeline(PCA())
        
    pca_pipe.fit(cuanti)
    modelo_pca = pca_pipe.named_steps['pca']

    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    
    
    
    pcaa = prince.PCA(
        n_components=len(cuanti.columns),
         n_iter=3,
         rescale_with_mean=corr,
         rescale_with_std=corr,
         copy=True,
         check_input=True,
         engine='auto'
     )
    pcaa = pcaa.fit(cuanti)
        
    resultados = pd.DataFrame()
    val_p = [round(i,3) for i in pcaa.eigenvalues_]
    var_exp = [round(i,3) for i in pcaa.explained_inertia_]
    var_ex_acum = []
    a=0
    for i in range(len(var_exp)):
        a += var_exp[i]
        var_ex_acum.append(a)
        
        
    
    resultados['Autovalores Propios'] = val_p
    resultados['Varianza Explicada'] = modelo_pca.explained_variance_ratio_
    resultados['Varianza Explicada Acumulada'] = prop_varianza_acum
    
        
    return resultados
    
