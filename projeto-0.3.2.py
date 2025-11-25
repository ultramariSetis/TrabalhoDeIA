import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

#importante: para alterar entre 2 var e 3 var, alterar as linhas 24/25 e 55/56

# define uma função para plotar clusters baseados em n e na categoria a ser considerada usand K-médias
def plt_cluster(k=3,cat='Annual Income (k$)'):
    km_n = KMeans(n_clusters=k, random_state=1).fit(X)
    X['Labels'] = km_n.labels_
    plt.figure(figsize=(8, 3))
    sns.scatterplot(x=X[cat], y=X['Spending Score (1-100)'], hue=X['Labels'], palette=sns.color_palette('hls', k))
    plt.title(f'KMeans with {k} Clusters - Spending Score (1-100) vs {cat}')
    plt.show()

# 1. Carregar o dataset
df = pd.read_csv('Mall_Customers.csv')  # Substitua pelo caminho do arquivo

# 2. Selecionar features relevantes
#features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

# 3. Pré-processamento: Normalização
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# 4. Determinar k com Elbow Method (opcional, para validação)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=1)   #Random State 42 é uma referencia, pode ser qualquer valor
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=range(1, 11), y=inertia, ax=ax, marker='o')
ax.set_title('Cluster SEQ')
ax.set_xlabel('Clusters')
ax.set_ylabel('SEQ')
plt.show()


#Comparar os graficos por genero
df.set_index('CustomerID', inplace=True)
sns.pairplot(df, hue='Genre', aspect=2)
plt.show()

# Descobrir o valor de K ideal
#n_lista = [3,4,5,6]  #Comentado para usar apenas quando for printar os outros k valores
n_lista = [5]
#cat_lista = ['Age','Annual Income (k$)']
cat_lista = ['Annual Income (k$)']
for n in n_lista:
    for cat in cat_lista:
        plt_cluster(n,cat)

# Isso vai mostrar que k = 5 é o ideal para comparar idade x renda
k=5
i=2

#Teste para ver diversos valores do Silhouette Score
for i in range(2,11):
    # 5. Aplicar K-Means com k=3
    kmeans = KMeans(n_clusters=i, random_state=42) #Originalmente k=3
    df['Cluster'] = kmeans.fit_predict(X)


    #Calcular o Silhouette Score
    sil_score = silhouette_score(X, df['Cluster'])
    print(f"Silhouette Score para k={i}: {sil_score}") # Originalmente k=k

#Voltando ao valor original de K
kmeans = KMeans(n_clusters=k, random_state=42) #Originalmente k=3
df['Cluster'] = kmeans.fit_predict(X)
sil_score = silhouette_score(X, df['Cluster'])
print(f"Silhouette Score para k={k}: {sil_score}") # Originalmente k=k


# 6. Visualização com PCA (redução para 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X) #Tem como melhorar o posicionamento dos centroides no PCA
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
plt.title('Clusters de Clientes (PCA)')
plt.legend(title='Cluster')
plt.show()

# 7. Análise dos clusters
cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)
#print(f"SEQ = {kmeans.inertia_}")
#print(f"SEQ = {inertia}")

j=1
for j in range(10):
    print(f"SEQ ({j+1}) = {inertia[j]}")