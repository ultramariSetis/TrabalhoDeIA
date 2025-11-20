import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# 1. Carregar o dataset
df = pd.read_csv('Mall_Customers.csv')  # Substitua pelo caminho do arquivo

# 2. Selecionar features relevantes
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

# 3. Pré-processamento: Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Determinar k com Elbow Method (opcional, para validação)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)   #Random State 42 é uma referencia, pode ser qualquer valor
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('SEQ')
plt.show()

# Descobrir o valor de K ideal
#n_lista = [3,4,5,6]  #Comentado para usar apenas quando for printar os outros k valores
n_lista = [5]
cat_lista = ['Age','Annual Income (k$)']
for n in n_lista:
    for cat in cat_lista:
        plt_cluster(n,cat)

# Isso vai mostrar que k = 5 é o ideal para comparar idade x renda


# 5. Aplicar K-Means com k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Visualização com PCA (redução para 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
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
print(f"SEQ = {kmeans.inertia_}")
print(f"SEQ = {inertia}")