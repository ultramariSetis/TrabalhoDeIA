import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# define uma função para plotar clusters baseados em n e na categoria a ser considerada usand K-médias
def plt_cluster(k=3,cat='Annual Income (k$)'):
    km_n = KMeans(n_clusters=k, random_state=1).fit(X)
    X['Labels'] = km_n.labels_
    plt.figure(figsize=(8, 3))
    sns.scatterplot(x=X[cat], y=X['Spending Score (1-100)'], hue=X['Labels'], palette=sns.color_palette('hls', k))
    plt.title(f'KMeans with {k} Clusters - Spending Score (1-100) vs {cat}')
    plt.show()

#Define o modo os parâmetros do programa

resp = input("Usar Age? S/N?")
if (resp == "S"):
    print("Vamos usar Age")
    cat_lista = ['Age','Annual Income (k$)']
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
else:
    print("Não vamos usar Age")
    cat_lista = ['Annual Income (k$)']
    features = ['Annual Income (k$)', 'Spending Score (1-100)']


# 1. Carregar o dataset
df = pd.read_csv('Mall_Customers.csv')  # Substitua pelo caminho do arquivo

# 2. Selecionar features relevantes
X = df[features].copy()

# 3. Determinar k com Elbow Method (opcional, para validação)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)   #Random State 42 é uma referencia, pode ser qualquer valor
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# 3.1. Plotar o Gráfico do Elbow
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=range(1, 11), y=inertia, ax=ax, marker='o')
ax.set_title('Cluster SEQ')
ax.set_xlabel('Clusters')
ax.set_ylabel('SEQ')
plt.show()


#4. Comparar os Gráficos por Gênero
df.set_index('CustomerID', inplace=True)
sns.pairplot(df, hue='Genre', aspect=2)
plt.show()

#5. Descobrir o valor de K ideal
#n_lista = [3,4,5,6]  #Comentado para usar apenas quando for printar os outros k valores
n_lista = [5]
for n in n_lista:
    for cat in cat_lista:
        plt_cluster(n,cat)

# Isso vai mostrar que k = 5 é o ideal para comparar idade x renda
k=5  #definimos k=5
i=2  

#6. Teste para ver diversos valores do Silhouette Score
for i in range(2,11):
    #7. Aplicar K-Means para k de 1 a k
    kmeans = KMeans(n_clusters=i, random_state=42)
    #df['Cluster'] = kmeans.fit_predict(X)
    labelx = kmeans.fit_predict(X)

    #8. Calcular o Silhouette Score
    sil_score = silhouette_score(X, labelx)
    print(f"Silhouette Score para k={i}: {sil_score}")

#9. Cálculo dos Clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 9.1. Análise dos clusters
cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)

#10. Analise dos valores de SEQ de 1 até 10
j=1
for j in range(10):
    print(f"SEQ ({j+1}) = {inertia[j]}")