# -*- coding: utf-8 -*-
"""
TESTE RÁPIDO: Otimização do K em KNeighborsClassifier
Testando K = 3, 5, 7, 10, 15
==================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TESTE RÁPIDO: OTIMIZAÇÃO DO K EM KNN")
print("=" * 70)

# ============================================================
# 1. Carregar e processar dados (reutilizando modelo_final.py)
# ============================================================
print("\n1. Carregando dados...")

try:
    df = pd.read_csv("Ibovespa.csv", encoding='utf-8')
except:
    df = pd.read_csv("Ibovespa.csv", encoding='latin1')

# Padronizar nomes de colunas
df.columns = ['Data', 'Ultimo', 'Abertura', 'Maxima', 'Minima', 'Vol', 'VarPerc']
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df = df.sort_values('Data').reset_index(drop=True)

# Processar valores numéricos
for col in ['Ultimo', 'Abertura', 'Maxima', 'Minima']:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Vol'] = df['Vol'].astype(str).str.replace('B', '', regex=False)
df['Vol'] = df['Vol'].str.replace(',', '.', regex=False)
df['Vol'] = pd.to_numeric(df['Vol'], errors='coerce')

df['VarPerc'] = df['VarPerc'].astype(str).str.replace('%', '', regex=False)
df['VarPerc'] = df['VarPerc'].str.replace(',', '.', regex=False)
df['VarPerc'] = pd.to_numeric(df['VarPerc'], errors='coerce')

df['Tendencia'] = (df['Ultimo'].shift(-1) > df['Ultimo']).astype(int)
df = df.dropna()

print(f"   ✓ Dados carregados: {len(df)} registros")

# ============================================================
# 2. Split treino/teste
# ============================================================
print("2. Separando treino/teste...")

split_point = len(df) - 30
df_train_raw = df.iloc[:split_point].copy()
df_test_raw = df.iloc[split_point:].copy()

print(f"   ✓ Treino: {len(df_train_raw)} | Teste: {len(df_test_raw)}")

# ============================================================
# 3. Criar features
# ============================================================
print("3. Criando features...")

def calcula_RSI(series, periodo=14):
    delta = series.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(periodo).mean()
    media_perda = perda.rolling(periodo).mean()
    RS = media_ganho / media_perda
    RSI = 100 - (100 / (1 + RS))
    return RSI

def calcula_MACD(series, curto=12, longo=26, sinal=9):
    EMA_curto = series.ewm(span=curto, adjust=False).mean()
    EMA_longo = series.ewm(span=longo, adjust=False).mean()
    MACD = EMA_curto - EMA_longo
    sinal_MACD = MACD.ewm(span=sinal, adjust=False).mean()
    return MACD, sinal_MACD

def criar_features(df_raw):
    df_feat = df_raw.copy()
    df_feat['Retorno'] = df_feat['Ultimo'].pct_change()
    df_feat['MM5'] = df_feat['Ultimo'].rolling(5).mean()
    df_feat['MM10'] = df_feat['Ultimo'].rolling(10).mean()
    df_feat['Volatilidade10'] = df_feat['Retorno'].rolling(10).std()
    df_feat['RSI14'] = calcula_RSI(df_feat['Ultimo'], periodo=14)
    df_feat['MACD'], df_feat['MACD_Sinal'] = calcula_MACD(df_feat['Ultimo'])
    return df_feat.dropna()

df_train = criar_features(df_train_raw)
df_test = criar_features(df_test_raw)

X_cols = ['Ultimo', 'Abertura', 'Maxima', 'Minima', 'Retorno', 'MM5', 'MM10', 
          'Volatilidade10', 'RSI14', 'MACD', 'MACD_Sinal']

X_train = df_train[X_cols]
y_train = df_train['Tendencia']
X_test = df_test[X_cols]
y_test = df_test['Tendencia']

print(f"   ✓ Features criadas: {len(X_cols)} indicadores")

# ============================================================
# 4. Normalizar
# ============================================================
print("4. Normalizando dados...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✓ StandardScaler fit em treino")

# ============================================================
# 5. Testar diferentes K
# ============================================================
print("\n5. Testando K = 3, 5, 7, 10, 15...\n")

K_values = [3, 5, 7, 10, 15]
results = []

for k in K_values:
    print(f"   Testando K = {k}...", end=" ")
    
    # Treinar
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    
    # Predições
    y_pred_train = knn.predict(X_train_scaled)
    y_proba_train = knn.predict_proba(X_train_scaled)[:, 1]
    
    y_pred_test = knn.predict(X_test_scaled)
    y_proba_test = knn.predict_proba(X_test_scaled)[:, 1]
    
    # Métricas treino
    acc_train = accuracy_score(y_train, y_pred_train)
    auc_train = roc_auc_score(y_train, y_proba_train)
    
    # Métricas teste
    acc_test = accuracy_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_proba_test)
    prec_test = precision_score(y_test, y_pred_test)
    rec_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    
    # Gap overfitting
    gap = acc_train - acc_test
    
    # Validação cruzada (5 folds rápido)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for train_idx, test_idx in tscv.split(X_train_scaled):
        X_fold_train = X_train_scaled[train_idx]
        X_fold_test = X_train_scaled[test_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_test = y_train.iloc[test_idx]
        
        knn_fold = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
        knn_fold.fit(X_fold_train, y_fold_train)
        
        y_pred_fold = knn_fold.predict(X_fold_test)
        acc_fold = accuracy_score(y_fold_test, y_pred_fold)
        cv_scores.append(acc_fold)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    results.append({
        'K': k,
        'Treino Acc': acc_train,
        'Teste Acc': acc_test,
        'Gap': gap,
        'AUC Train': auc_train,
        'AUC Test': auc_test,
        'Precisão': prec_test,
        'Recall': rec_test,
        'F1-Score': f1_test,
        'CV Mean': cv_mean,
        'CV Std': cv_std
    })
    
    print(f"✓ (Acc: {acc_test:.1%}, AUC: {auc_test:.3f}, Gap: {gap:.1%})")

# ============================================================
# 6. Exibir resultados
# ============================================================
print("\n" + "=" * 70)
print("RESULTADOS DETALHADOS")
print("=" * 70 + "\n")

results_df = pd.DataFrame(results)

# Tabela 1: Performance Teste
print("📊 PERFORMANCE NO TESTE:\n")
table1 = results_df[['K', 'Teste Acc', 'AUC Test', 'Precisão', 'Recall', 'F1-Score']].copy()
table1['Teste Acc'] = table1['Teste Acc'].apply(lambda x: f"{x:.1%}")
table1['AUC Test'] = table1['AUC Test'].apply(lambda x: f"{x:.4f}")
table1['Precisão'] = table1['Precisão'].apply(lambda x: f"{x:.4f}")
table1['Recall'] = table1['Recall'].apply(lambda x: f"{x:.4f}")
table1['F1-Score'] = table1['F1-Score'].apply(lambda x: f"{x:.4f}")
print(table1.to_string(index=False))

# Tabela 2: Overfitting & CV
print("\n\n📈 GENERALIZAÇÃO & VALIDAÇÃO CRUZADA:\n")
table2 = results_df[['K', 'Treino Acc', 'Teste Acc', 'Gap', 'CV Mean', 'CV Std']].copy()
table2['Treino Acc'] = table2['Treino Acc'].apply(lambda x: f"{x:.1%}")
table2['Teste Acc'] = table2['Teste Acc'].apply(lambda x: f"{x:.1%}")
table2['Gap'] = table2['Gap'].apply(lambda x: f"{x:.1%}")
table2['CV Mean'] = table2['CV Mean'].apply(lambda x: f"{x:.1%}")
table2['CV Std'] = table2['CV Std'].apply(lambda x: f"±{x:.1%}")
print(table2.to_string(index=False))

# ============================================================
# 7. Identificar K ótimo
# ============================================================
print("\n\n" + "=" * 70)
print("RECOMENDAÇÕES")
print("=" * 70)

best_acc = results_df.loc[results_df['Teste Acc'].idxmax()]
best_auc = results_df.loc[results_df['AUC Test'].idxmax()]
best_gap = results_df.loc[results_df['Gap'].idxmin()]
best_cv = results_df.loc[results_df['CV Mean'].idxmax()]

print(f"""
✓ MELHOR ACURÁCIA TESTE: K = {int(best_acc['K'])}
  Acurácia: {best_acc['Teste Acc']:.1%}
  AUC: {best_acc['AUC Test']:.4f}
  Gap: {best_acc['Gap']:.1%}

✓ MELHOR AUC: K = {int(best_auc['K'])}
  AUC: {best_auc['AUC Test']:.4f}
  Acurácia: {best_auc['Teste Acc']:.1%}
  Gap: {best_auc['Gap']:.1%}

✓ MELHOR GENERALIZAÇÃO (Menor Gap): K = {int(best_gap['K'])}
  Gap: {best_gap['Gap']:.1%}
  Acurácia: {best_gap['Teste Acc']:.1%}
  CV Mean: {best_gap['CV Mean']:.1%}

✓ MELHOR CV SCORE: K = {int(best_cv['K'])}
  CV Mean: {best_cv['CV Mean']:.1%}
  Acurácia: {best_cv['Teste Acc']:.1%}
""")

# ============================================================
# 8. Gráficos comparativos
# ============================================================
print("Gerando gráficos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Acurácia Treino vs Teste
ax = axes[0, 0]
x_pos = np.arange(len(K_values))
width = 0.35
ax.bar(x_pos - width/2, results_df['Treino Acc'], width, label='Treino', alpha=0.8, color='#2E86AB')
ax.bar(x_pos + width/2, results_df['Teste Acc'], width, label='Teste', alpha=0.8, color='#A23B72')
ax.set_xlabel('K', fontweight='bold')
ax.set_ylabel('Acurácia', fontweight='bold')
ax.set_title('Acurácia: Treino vs Teste', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(K_values)
ax.legend()
ax.grid(True, alpha=0.3)
for i, (train, test) in enumerate(zip(results_df['Treino Acc'], results_df['Teste Acc'])):
    ax.text(i - width/2, train + 0.02, f'{train:.1%}', ha='center', fontsize=9)
    ax.text(i + width/2, test + 0.02, f'{test:.1%}', ha='center', fontsize=9)

# Gráfico 2: AUC Treino vs Teste
ax = axes[0, 1]
ax.plot(K_values, results_df['AUC Train'], marker='o', linewidth=2.5, markersize=8, 
        label='Treino', color='#2E86AB', alpha=0.7)
ax.plot(K_values, results_df['AUC Test'], marker='s', linewidth=2.5, markersize=8, 
        label='Teste', color='#A23B72', alpha=0.7)
ax.set_xlabel('K', fontweight='bold')
ax.set_ylabel('AUC-ROC', fontweight='bold')
ax.set_title('AUC: Treino vs Teste', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(K_values)

# Gráfico 3: Gap Overfitting
ax = axes[1, 0]
colors = ['#06A77D' if g < 0.10 else '#FFB703' if g < 0.20 else '#D62828' for g in results_df['Gap']]
bars = ax.bar(K_values, results_df['Gap'], color=colors, alpha=0.8)
ax.set_xlabel('K', fontweight='bold')
ax.set_ylabel('Gap (%)', fontweight='bold')
ax.set_title('Overfitting Gap (Treino - Teste)', fontweight='bold')
ax.axhline(y=0.10, color='blue', linestyle='--', alpha=0.5, label='Limite OK (10%)')
ax.axhline(y=0.20, color='orange', linestyle='--', alpha=0.5, label='Limite Moderado (20%)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(K_values)
for i, gap in enumerate(results_df['Gap']):
    ax.text(K_values[i], gap + 0.01, f'{gap:.1%}', ha='center', fontsize=9, fontweight='bold')

# Gráfico 4: Cross-Validation Score
ax = axes[1, 1]
ax.errorbar(K_values, results_df['CV Mean'], yerr=results_df['CV Std'], 
            marker='o', linewidth=2, markersize=8, capsize=5, color='#F18F01', alpha=0.8)
ax.fill_between(K_values, 
                results_df['CV Mean'] - results_df['CV Std'],
                results_df['CV Mean'] + results_df['CV Std'],
                alpha=0.2, color='#F18F01')
ax.set_xlabel('K', fontweight='bold')
ax.set_ylabel('CV Mean Accuracy', fontweight='bold')
ax.set_title('Cross-Validation Score (5 Folds)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(K_values)
for k, cv, std in zip(K_values, results_df['CV Mean'], results_df['CV Std']):
    ax.text(k, cv + std + 0.02, f'{cv:.1%}', ha='center', fontsize=9)

plt.suptitle('Otimização de K em KNeighborsClassifier', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('teste_knn_k_otimo.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 'teste_knn_k_otimo.png'")
plt.show()

# ============================================================
# 9. Salvar resultados
# ============================================================
csv_filename = 'resultados_knn_k_otimo.csv'
results_df.to_csv(csv_filename, index=False)
print(f"✓ Resultados salvos: '{csv_filename}'")

print("\n" + "=" * 70)
print("TESTE CONCLUÍDO")
print("=" * 70)
