"""
Projeto: Segmentação de Clientes com Autoencoder + KMeans
Base: Mall Customers (Mall_Customers.csv)
Como usar:
 - Coloque 'Mall_Customers.csv' na mesma pasta ou carregue no Colab.
 - Instale dependências: pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# --- Reprodutibilidade ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# --- Configurações (ajuste conforme necessário) ---
CSV_PATH = "Mall_Customers.csv"   # caminho para o CSV
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
USE_GENDER = True                 # incluir Gender como feature binária?
LATENT_DIM = 3                    # dimensão do bottleneck do autoencoder
BATCH_SIZE = 16
EPOCHS = 200
VALIDATION_SPLIT = 0.2
N_CLUSTERS = 5                    # número inicial de clusters KMeans
PLOT_SAVE_DIR = "outputs"

os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# --- 1. Carregar dados ---
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Arquivo '{CSV_PATH}' não encontrado. Coloque o CSV no diretório ou ajuste CSV_PATH.")
df = pd.read_csv(CSV_PATH)

# Mostra as primeiras linhas
print("Preview dos dados:")
print(df.head())

# --- 2. Pré-processamento ---
data = df.copy()
le = LabelEncoder()

# Opcional: codificar gênero
if USE_GENDER and 'Genre' in data.columns:
    data['Gender_bin'] = le.fit_transform(data['Genre'])
    feature_cols = FEATURES + ['Gender_bin']
else:
    feature_cols = FEATURES

# Seleciona apenas as features desejadas
X = data[feature_cols].copy()
# Renomeia colunas problemáticas
X.columns = [c.replace(" ", "_").replace("(", "").replace(")", "").replace("–","-") for c in X.columns]

# Verificação de NA
if X.isnull().any().any():
    print("Existem NAs — preenchendo com mediana.")
    X = X.fillna(X.median())

# Escalonamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures usadas:", feature_cols)
print("Shape do dataset (após escalonamento):", X_scaled.shape)

# --- 3. Construção do Autoencoder ---
input_dim = X_scaled.shape[1]
encoding_dim = LATENT_DIM

# Encoder
input_layer = layers.Input(shape=(input_dim,))
x = layers.Dense(64, activation='relu')(input_layer)
x = layers.Dense(32, activation='relu')(x)
encoded = layers.Dense(encoding_dim, activation='linear', name='bottleneck')(x)

# Decoder
x = layers.Dense(32, activation='relu')(encoded)
x = layers.Dense(64, activation='relu')(x)
decoded = layers.Dense(input_dim, activation='linear')(x)

autoencoder = models.Model(inputs=input_layer, outputs=decoded, name='autoencoder')
encoder = models.Model(inputs=input_layer, outputs=encoded, name='encoder')

autoencoder.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mse')

autoencoder.summary()

# --- 4. Treino do Autoencoder ---
es = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[es],
    verbose=2
)

# salva histórico de treino (plot)
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Autoencoder Loss')
plt.xlabel('epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_SAVE_DIR, 'autoencoder_loss.png'))
plt.show()

# --- 5. Extração do Espaço Latente (Embeddings) ---
embeddings = encoder.predict(X_scaled)
print("Shape do espaço latente:", embeddings.shape)

# Se LATENT_DIM > 2, podemos reduzir para 2D apenas para visualização (PCA)
if embeddings.shape[1] > 2:
    pca_vis = PCA(n_components=2, random_state=SEED)
    emb_2d = pca_vis.fit_transform(embeddings)
else:
    emb_2d = embeddings

# --- 6. Clusterização com KMeans no espaço latente ---
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20)
clusters = kmeans.fit_predict(embeddings)

# Métrica de qualidade
sil = silhouette_score(embeddings, clusters)
print(f"Silhouette Score (latent space) para k={N_CLUSTERS}: {sil:.4f}")

# Anexa resultados ao dataframe original
data['cluster'] = clusters
data['embed_1'] = emb_2d[:,0]
data['embed_2'] = emb_2d[:,1]

# --- 7. Visualizações ---
plt.figure(figsize=(8,6))
sns.scatterplot(x='embed_1', y='embed_2', hue='cluster', palette='tab10', data=data, s=60)
plt.title(f'Clusters no espaço latente (k={N_CLUSTERS})')
plt.legend(title='cluster', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_SAVE_DIR, 'clusters_latent_space.png'))
plt.show()

# Plot das features reais com clusters
pair_cols = X.columns.tolist()
if len(pair_cols) >= 2:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=pair_cols[0], y=pair_cols[1], hue='cluster', data=pd.concat([X.reset_index(drop=True), data['cluster']], axis=1), s=60)
    plt.title(f'Clusters em {pair_cols[0]} x {pair_cols[1]}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'clusters_feature_space.png'))
    plt.show()

# Estatísticas por cluster
cluster_summary = data.groupby('cluster')[feature_cols].agg(['count', 'mean', 'median', 'std'])
print("\nResumo por cluster (algumas estatísticas):")
print(cluster_summary)

# Avaliar cada cluster por estatística
summary = df.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(summary)

# Salva resultados
data.to_csv(os.path.join(PLOT_SAVE_DIR, 'mall_customers_with_clusters.csv'), index=False)
encoder.save(os.path.join(PLOT_SAVE_DIR, 'encoder_model.h5'))
autoencoder.save(os.path.join(PLOT_SAVE_DIR, 'autoencoder_model.h5'))

print(f"\nResultados e modelos salvos em: {PLOT_SAVE_DIR}")
