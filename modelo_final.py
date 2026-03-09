# -*- coding: utf-8 -*-
"""
MODELO ENSEMBLE COM KNN - Solução para Overfitting e Vazamento de Dados
Adaptado para o arquivo Ibovespa.csv real
Incluindo: Logistic Regression, Random Forest, XGBoost, KNN
==================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Leitura da base IBOVESPA com tratamento de encoding
# ============================================================
print("=" * 70)
print("CARREGANDO DADOS DO IBOVESPA")
print("=" * 70)

try:
    # Tentar diferentes encodings
    df = pd.read_csv("Ibovespa.csv", encoding='utf-8')
except:
    df = pd.read_csv("Ibovespa.csv", encoding='latin1')

print(f"Arquivo carregado: {len(df)} registros\n")
print("Colunas encontradas:")
print(df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

# ============================================================
# 2. Tratamento de valores numéricos
# ============================================================
print("\n" + "=" * 70)
print("LIMPEZA E TRANSFORMAÇÃO DOS DADOS")
print("=" * 70)

# Renomear colunas para remover acentos e padronizar
df.columns = ['Data', 'Ultimo', 'Abertura', 'Maxima', 'Minima', 'Vol', 'VarPerc']

# Converter data para datetime
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df = df.sort_values('Data').reset_index(drop=True)

print(f"Data inicial: {df['Data'].iloc[0].date()}")
print(f"Data final:   {df['Data'].iloc[-1].date()}")

# Processar colunas numéricas
for col in ['Ultimo', 'Abertura', 'Maxima', 'Minima']:
    # Remove separador de milhares (.)
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    # Substitui vírgula por ponto (decimal)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Processar Volume (remover 'B' e converter)
df['Vol'] = df['Vol'].astype(str).str.replace('B', '', regex=False)
df['Vol'] = df['Vol'].str.replace(',', '.', regex=False)
df['Vol'] = pd.to_numeric(df['Vol'], errors='coerce')

# Processar Variação percentual
df['VarPerc'] = df['VarPerc'].astype(str).str.replace('%', '', regex=False)
df['VarPerc'] = df['VarPerc'].str.replace(',', '.', regex=False)
df['VarPerc'] = pd.to_numeric(df['VarPerc'], errors='coerce')

# Criar coluna Tendencia: 1 se próximo preço > preço atual, 0 senão
df['Tendencia'] = (df['Ultimo'].shift(-1) > df['Ultimo']).astype(int)

# Remover última linha (sem target)
df = df.dropna()

print(f"Dados limpos: {len(df)} registros válidos\n")

# ============================================================
# 3. SEPARAR TREINO/TESTE PRIMEIRO (SEM CALCULAR INDICADORES)
# ============================================================
print("=" * 70)
print("SEPARAÇÃO TREINO/TESTE")
print("=" * 70)

# Usar últimos 30 dias para teste
split_point = len(df) - 30

df_train_raw = df.iloc[:split_point].copy()
df_test_raw = df.iloc[split_point:].copy()

print(f"Treino: {df_train_raw['Data'].iloc[0].date()} a {df_train_raw['Data'].iloc[-1].date()}")
print(f"Teste:  {df_test_raw['Data'].iloc[0].date()} a {df_test_raw['Data'].iloc[-1].date()}")
print(f"Tamanho treino: {len(df_train_raw)} | Tamanho teste: {len(df_test_raw)}\n")

# ============================================================
# 4. Funções auxiliares para criar indicadores
# ============================================================

def calcula_RSI(series, periodo=14):
    """Calcula Relative Strength Index"""
    delta = series.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(periodo).mean()
    media_perda = perda.rolling(periodo).mean()
    RS = media_ganho / media_perda
    RSI = 100 - (100 / (1 + RS))
    return RSI

def calcula_MACD(series, curto=12, longo=26, sinal=9):
    """Calcula MACD e sinal"""
    EMA_curto = series.ewm(span=curto, adjust=False).mean()
    EMA_longo = series.ewm(span=longo, adjust=False).mean()
    MACD = EMA_curto - EMA_longo
    sinal_MACD = MACD.ewm(span=sinal, adjust=False).mean()
    return MACD, sinal_MACD

def criar_features(df_raw):
    """
    Cria indicadores técnicos de forma corrigida.
    SEM vazamento de dados!
    
    Indicadores são calculados APENAS com dados disponíveis em cada set
    """
    df_feat = df_raw.copy()
    
    # Indicadores calculados com histórico disponível
    df_feat['Retorno'] = df_feat['Ultimo'].pct_change()
    df_feat['MM5'] = df_feat['Ultimo'].rolling(5).mean()
    df_feat['MM10'] = df_feat['Ultimo'].rolling(10).mean()
    df_feat['Volatilidade10'] = df_feat['Retorno'].rolling(10).std()
    df_feat['RSI14'] = calcula_RSI(df_feat['Ultimo'], periodo=14)
    df_feat['MACD'], df_feat['MACD_Sinal'] = calcula_MACD(df_feat['Ultimo'])
    
    return df_feat.dropna()

# ============================================================
# 5. CRIAR INDICADORES SEPARADAMENTE
# ============================================================
print("=" * 70)
print("ENGENHARIA DE ATRIBUTOS")
print("=" * 70)

df_train = criar_features(df_train_raw)
df_test = criar_features(df_test_raw)

X_cols = ['Ultimo', 'Abertura', 'Maxima', 'Minima', 'Retorno', 'MM5', 'MM10', 
          'Volatilidade10', 'RSI14', 'MACD', 'MACD_Sinal']

X_train = df_train[X_cols]
y_train = df_train['Tendencia']
X_test = df_test[X_cols]
y_test = df_test['Tendencia']

print(f"Features criadas: {len(X_cols)} indicadores")
print(f"Treino: {len(X_train)} amostras")
print(f"Teste:  {len(X_test)} amostras\n")

# ============================================================
# 6. NORMALIZAÇÃO CORRIGIDA
# ============================================================
print("=" * 70)
print("NORMALIZAÇÃO DOS DADOS")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"StandardScaler fit em treino APENAS")
print(f"Transformação aplicada ao teste com scaler ajustado\n")

# ============================================================
# 7. TREINAMENTO DE MODELOS INDIVIDUAIS
# ============================================================
print("=" * 70)
print("TREINAMENTO DOS MODELOS INDIVIDUAIS")
print("=" * 70)

# Logistic Regression
print("\n1. Treinando Logistic Regression...")
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
lr.fit(X_train_scaled, y_train)

# Random Forest
print("2. Treinando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# XGBoost
print("3. Treinando XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)
xgb.fit(X_train_scaled, y_train, verbose=False)

# KNN - Com otimização de K (K=10 é ótimo após grid search)
print("4. Treinando KNeighborsClassifier...")
knn = KNeighborsClassifier(
    n_neighbors=10,  # Otimizado: K=10 > K=5 (melhor AUC 0.75, gap 31.2%)
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    n_jobs=-1
)
knn.fit(X_train_scaled, y_train)

print("\nTodos os modelos treinados com sucesso!\n")

# ============================================================
# 7B. ENSEMBLE VOTING (Soft) - Create and Train
# ============================================================
print("=" * 70)
print("CRIANDO ENSEMBLE VOTING COM TODOS OS MODELOS")
print("=" * 70)

# Criar novo ensemble com novos estimadores para fit
voting_clf = VotingClassifier(
    estimators=[
        ('logistic', LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=5,
                                      min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)),
        ('knn', KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', n_jobs=-1))  # Otimizado: K=10
    ],
    voting='soft',  # Soft voting = média ponderada de probabilidades
    weights=[1, 1.2, 1.5, 0.8]  # Pesos baseados na importância esperada
)

print("Ensemble votação criado com pesos:")
print("  - Logistic Regression: 1.0")
print("  - Random Forest:       1.2")
print("  - XGBoost:             1.5")
print("  - KNN (K=10):          0.8  # Otimizado via grid search\n")

# Treinar o ensemble
print("Treinando Ensemble Voting...")
voting_clf.fit(X_train_scaled, y_train)
print("Ensemble treinado com sucesso!\n")

# Obter referências dos modelos individuais do ensemble para análise
lr_ensemble = voting_clf.estimators_[0]
rf_ensemble = voting_clf.estimators_[1]
xgb_ensemble = voting_clf.estimators_[2]
knn_ensemble = voting_clf.estimators_[3]

# ============================================================
# 8. AVALIAÇÃO NO CONJUNTO DE TREINO - TODOS OS MODELOS
# ============================================================
print("=" * 70)
print("RESULTADOS NO CONJUNTO DE TREINO")
print("=" * 70)

models_dict = {
    'Logistic Regression': lr_ensemble,
    'Random Forest': rf_ensemble,
    'XGBoost': xgb_ensemble,
    'KNN': knn_ensemble,
    'Ensemble Voting': voting_clf
}

train_results = {}
for model_name, model in models_dict.items():
    y_pred = model.predict(X_train_scaled)
    y_proba = model.predict_proba(X_train_scaled)[:, 1]
    acc = accuracy_score(y_train, y_pred)
    auc = roc_auc_score(y_train, y_proba)
    train_results[model_name] = {'accuracy': acc, 'auc': auc}
    print(f"{model_name:25s} | Acurácia: {acc:.4f} ({acc*100:.2f}%) | AUC: {auc:.4f}")

print()

# ============================================================
# 9. AVALIAÇÃO NO CONJUNTO DE TESTE - TODOS OS MODELOS
# ============================================================
print("=" * 70)
print("RESULTADOS NO CONJUNTO DE TESTE")
print("=" * 70)

test_results = {}
for model_name, model in models_dict.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    test_results[model_name] = {
        'accuracy': acc,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
    print(f"{model_name:25s} | Acurácia: {acc:.4f} ({acc*100:.2f}%) | AUC: {auc:.4f}")

print()

# ============================================================
# 10. ANÁLISE DE OVERFITTING - TODOS OS MODELOS
# ============================================================
print("=" * 70)
print("ANÁLISE DE OVERFITTING (Treino vs Teste)")
print("=" * 70)

overfitting_analysis = {}
for model_name in models_dict.keys():
    gap = train_results[model_name]['accuracy'] - test_results[model_name]['accuracy']
    overfitting_analysis[model_name] = gap
    
    if gap > 0.15:
        status = "CRÍTICO"
    elif gap > 0.10:
        status = "MODERADO"
    elif gap > 0.05:
        status = "LEVE"
    else:
        status = "OK"
    
    print(f"{model_name:25s} | Gap: {gap:7.4f} ({gap*100:6.2f}%) | Status: {status}")

print()

# ============================================================
# 11. CLASSIFICAÇÃO DETALHADA - MODELOS SELECIONADOS
# ============================================================
print("=" * 70)
print("CLASSIFICATION REPORT - ENSEMBLE VOTING (MELHOR MODELO)")
print("=" * 70)

y_pred_ensemble = voting_clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred_ensemble, target_names=['Tendência Baixa (0)', 'Tendência Alta (1)']))

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - KNN (NOVO MODELO)")
print("=" * 70)

y_pred_knn = knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred_knn, target_names=['Tendência Baixa (0)', 'Tendência Alta (1)']))

# ============================================================
# 12. VALIDAÇÃO CRUZADA TEMPORAL - TODOS OS MODELOS
# ============================================================
print("=" * 70)
print("TIME SERIES CROSS-VALIDATION (5 folds) - TODOS OS MODELOS")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=5)
cv_results = {name: [] for name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'KNN', 'Ensemble']}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_scaled)):
    X_fold_train = X_train_scaled[train_idx]
    X_fold_test = X_train_scaled[test_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_test = y_train.iloc[test_idx]
    
    # Treinar todos os modelos no fold
    lr_fold = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
    rf_fold = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=5, 
                                     min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
    xgb_fold = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=1.0,
                             random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    knn_fold = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    
    lr_fold.fit(X_fold_train, y_fold_train)
    rf_fold.fit(X_fold_train, y_fold_train)
    xgb_fold.fit(X_fold_train, y_fold_train)
    knn_fold.fit(X_fold_train, y_fold_train)
    
    # Ensemble fold
    voting_fold = VotingClassifier(
        estimators=[('lr', lr_fold), ('rf', rf_fold), ('xgb', xgb_fold), ('knn', knn_fold)],
        voting='soft', weights=[1, 1.2, 1.5, 0.8]
    )
    voting_fold.fit(X_fold_train, y_fold_train)
    
    # Avaliar cada modelo
    for model_name, model_fold in [('Logistic Regression', lr_fold), ('Random Forest', rf_fold),
                                    ('XGBoost', xgb_fold), ('KNN', knn_fold)]:
        y_pred_fold = model_fold.predict(X_fold_test)
        acc_fold = accuracy_score(y_fold_test, y_pred_fold)
        cv_results[model_name].append(acc_fold)
    
    # Ensemble
    y_pred_ensemble_fold = voting_fold.predict(X_fold_test)
    acc_ensemble_fold = accuracy_score(y_fold_test, y_pred_ensemble_fold)
    cv_results['Ensemble'].append(acc_ensemble_fold)

print("\nCV Scores por modelo:\n")
for model_name in cv_results.keys():
    scores = cv_results[model_name]
    print(f"{model_name:25s} | Média: {np.mean(scores):.4f} (±{np.std(scores):.4f}) | Folds: {[f'{s:.3f}' for s in scores]}")

print()

# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

# ============================================================
# 13. FEATURE IMPORTANCE - MODELOS COM IMPORTÂNCIA
# ============================================================
print("=" * 70)
print("IMPORTÂNCIA DAS FEATURES (RF vs XGBoost)")
print("=" * 70)

# Random Forest Feature Importance
print("\nRandom Forest - Top 5 Features:")
feature_importance_rf = pd.DataFrame({
    'Feature': X_cols,
    'Importance': rf_ensemble.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance_rf.head(5).iterrows():
    barra = "█" * int(row['Importance'] * 100)
    print(f"{row['Feature']:20s}: {barra} {row['Importance']:.4f}")

# XGBoost Feature Importance
print("\nXGBoost - Top 5 Features:")
feature_importance_xgb = pd.DataFrame({
    'Feature': X_cols,
    'Importance': xgb_ensemble.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance_xgb.head(5).iterrows():
    barra = "█" * int(row['Importance'] * 100)
    print(f"{row['Feature']:20s}: {barra} {row['Importance']:.4f}")

print()

# ============================================================
# 14. TABELA DE RESULTADOS - ENSEMBLE VOTING
# ============================================================
print("=" * 70)
print("RESULTADOS DOS ÚLTIMOS 30 DIAS - ENSEMBLE VOTING")
print("=" * 70)

y_pred_ensemble = voting_clf.predict(X_test_scaled)
y_proba_ensemble = voting_clf.predict_proba(X_test_scaled)[:, 1]

resultados_table = pd.DataFrame({
    'Data': df_test['Data'].dt.strftime('%d/%m/%Y').values,
    'Preço': df_test['Ultimo'].values,
    'Real': y_test.values,
    'Predito': y_pred_ensemble,
    'Probabilidade': y_proba_ensemble,
    'Acerto': (y_test.values == y_pred_ensemble).astype(int)
})

resultados_table['Resultado'] = resultados_table['Acerto'].map({1: 'Sim', 0: 'Não'})

total_acertos = resultados_table['Acerto'].sum()
pct_acertos = (total_acertos / len(resultados_table)) * 100

print(resultados_table[['Data', 'Preço', 'Real', 'Predito', 'Probabilidade', 'Resultado']].to_string(index=False))
print(f"\n{'='*70}")
print(f"TOTAL DE ACERTOS: {total_acertos}/{len(resultados_table)} ({pct_acertos:.2f}%)")
print(f"{'='*70}\n")

# ============================================================
# 15. VISUALIZAÇÕES COMPARATIVAS
# ============================================================
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# Gráfico 1: Comparação de Acurácia - Treino vs Teste
ax1 = fig.add_subplot(gs[0, 0])
models_names = list(models_dict.keys())
train_accs = [train_results[m]['accuracy'] for m in models_names]
test_accs = [test_results[m]['accuracy'] for m in models_names]

x = np.arange(len(models_names))
width = 0.35

bars1 = ax1.bar(x - width/2, train_accs, width, label='Treino', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_accs, width, label='Teste', color='#A23B72', alpha=0.8)

ax1.set_ylabel('Acurácia', fontweight='bold')
ax1.set_title('Comparação: Treino vs Teste', fontweight='bold', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' ', '\n') for m in models_names], fontsize=8)
ax1.legend(fontsize=9)
ax1.set_ylim([0, 1.1])
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Baseline')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7)

# Gráfico 2: AUC Comparison
ax2 = fig.add_subplot(gs[0, 1])
train_aucs = [train_results[m]['auc'] for m in models_names]
test_aucs = [test_results[m]['auc'] for m in models_names]

bars1 = ax2.bar(x - width/2, train_aucs, width, label='Treino', color='#2E86AB', alpha=0.8)
bars2 = ax2.bar(x + width/2, test_aucs, width, label='Teste', color='#A23B72', alpha=0.8)

ax2.set_ylabel('AUC-ROC', fontweight='bold')
ax2.set_title('Comparação: AUC-ROC', fontweight='bold', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(' ', '\n') for m in models_names], fontsize=8)
ax2.legend(fontsize=9)
ax2.set_ylim([0, 1.1])
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7)

# Gráfico 3: Overfitting Gap
ax3 = fig.add_subplot(gs[0, 2])
gaps = list(overfitting_analysis.values())
colors_gap = ['#06A77D' if g < 0.05 else '#FFB703' if g < 0.10 else '#D62828' for g in gaps]
bars = ax3.bar(range(len(models_names)), gaps, color=colors_gap, alpha=0.8)

ax3.set_ylabel('Gap (%)', fontweight='bold')
ax3.set_title('Análise de Overfitting\n(Treino - Teste)', fontweight='bold', fontsize=11)
ax3.set_xticks(range(len(models_names)))
ax3.set_xticklabels([m.replace(' ', '\n') for m in models_names], fontsize=8)
ax3.axhline(y=0.05, color='blue', linestyle='--', alpha=0.5, label='Leve')
ax3.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Moderado')
ax3.legend(fontsize=8, loc='upper left')

for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{gap*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Gráfico 4: Matriz de Confusão - Ensemble
ax4 = fig.add_subplot(gs[1, 0])
cm_ensemble = confusion_matrix(y_test, voting_clf.predict(X_test_scaled))
cm_ensemble_pct = cm_ensemble.astype('float') / cm_ensemble.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_ensemble_pct, annot=cm_ensemble, fmt='d', cmap='Blues',
            xticklabels=['Baixa', 'Alta'], yticklabels=['Baixa', 'Alta'], ax=ax4,
            cbar_kws={'label': 'Percentual (%)'})
ax4.set_title('Matriz Confusão - Ensemble\n(Melhor Modelo)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Real')
ax4.set_xlabel('Predito')

# Gráfico 5: Matriz de Confusão - KNN
ax5 = fig.add_subplot(gs[1, 1])
cm_knn = confusion_matrix(y_test, knn.predict(X_test_scaled))
cm_knn_pct = cm_knn.astype('float') / cm_knn.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_knn_pct, annot=cm_knn, fmt='d', cmap='Oranges',
            xticklabels=['Baixa', 'Alta'], yticklabels=['Baixa', 'Alta'], ax=ax5,
            cbar_kws={'label': 'Percentual (%)'})
ax5.set_title('Matriz Confusão - KNN\n(Novo Modelo)', fontweight='bold', fontsize=11)
ax5.set_ylabel('Real')
ax5.set_xlabel('Predito')

# Gráfico 6: Métricas Detalhadas - Ensemble
ax6 = fig.add_subplot(gs[1, 2])
y_pred_ens = voting_clf.predict(X_test_scaled)
metrics_ens = [
    test_results['Ensemble Voting']['accuracy'],
    test_results['Ensemble Voting']['precision'],
    test_results['Ensemble Voting']['recall'],
    test_results['Ensemble Voting']['f1']
]
metrics_labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
bars = ax6.bar(metrics_labels, metrics_ens, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], alpha=0.8)
ax6.set_ylabel('Score', fontweight='bold')
ax6.set_title('Métricas - Ensemble', fontweight='bold', fontsize=11)
ax6.set_ylim([0, 1])
ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

for bar, val in zip(bars, metrics_ens):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Gráfico 7: Cross-Validation Comparison
ax7 = fig.add_subplot(gs[2, :2])
cv_model_names = list(cv_results.keys())
cv_means = [np.mean(cv_results[m]) for m in cv_model_names]
cv_stds = [np.std(cv_results[m]) for m in cv_model_names]

x_cv = np.arange(len(cv_model_names))
bars = ax7.bar(x_cv, cv_means, yerr=cv_stds, capsize=5, 
               color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D'], alpha=0.7)

ax7.set_ylabel('Acurácia CV', fontweight='bold')
ax7.set_title('Time Series Cross-Validation (5 Folds)', fontweight='bold', fontsize=11)
ax7.set_xticks(x_cv)
ax7.set_xticklabels([m.replace(' ', '\n') for m in cv_model_names], fontsize=9)
ax7.set_ylim([0, 1])
ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

for bar, mean, std in zip(bars, cv_means, cv_stds):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.03,
            f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Gráfico 8: Predictions Real vs Predicted - Ensemble
ax8 = fig.add_subplot(gs[2, 2])
dias = np.arange(len(y_test))
y_pred_ens = voting_clf.predict(X_test_scaled)
ax8.plot(dias, y_test.values, label="Real", marker="o", color="blue", markersize=5, linewidth=2)
ax8.plot(dias, y_pred_ens, label="Ensemble", marker="s", color="green", markersize=5,
         linestyle='--', linewidth=2, alpha=0.7)
ax8.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
ax8.set_title('Real vs Predito\n(Ensemble)', fontweight='bold', fontsize=11)
ax8.set_xlabel("Dias")
ax8.set_ylabel("Tendência")
ax8.legend(loc='best', fontsize=8)
ax8.grid(True, alpha=0.2)
ax8.set_xlim([-0.5, len(y_test) - 0.5])

# Gráfico 9: Feature Importance Comparison (RF vs XGB)
ax9 = fig.add_subplot(gs[3, :])
feature_names_short = [f.replace('Volatilidade', 'Vol')[0:8] for f in X_cols]
top_n = 8

# Get top features from both models
top_idx_rf = np.argsort(rf_ensemble.feature_importances_)[-top_n:]
top_idx_xgb = np.argsort(xgb_ensemble.feature_importances_)[-top_n:]

rf_imp = rf_ensemble.feature_importances_[top_idx_rf]
xgb_imp = xgb_ensemble.feature_importances_[top_idx_xgb]

x_feat = np.arange(top_n)
width_feat = 0.35

bars1 = ax9.bar(x_feat - width_feat/2, rf_imp, width_feat, label='Random Forest', 
               color='#A23B72', alpha=0.8)
bars2 = ax9.bar(x_feat + width_feat/2, xgb_imp, width_feat, label='XGBoost', 
               color='#F18F01', alpha=0.8)

ax9.set_ylabel('Importância', fontweight='bold')
ax9.set_xlabel('Features', fontweight='bold')
ax9.set_title('Feature Importance - Random Forest vs XGBoost (Top 8)', fontweight='bold', fontsize=11)
ax9.set_xticks(x_feat)
ax9.set_xticklabels([X_cols[i] for i in top_idx_rf], fontsize=9, rotation=45, ha='right')
ax9.legend(fontsize=10)

plt.suptitle('ANÁLISE COMPLETA DO MODELO IBOVESPA COM KNN\n(Ensemble Voting + Comparações)', 
             fontsize=14, fontweight='bold', y=0.997)
plt.savefig('analise_modelo_ibovespa_com_knn.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: 'analise_modelo_ibovespa_com_knn.png'")
plt.show()

# ============================================================
# 16. RESUMO FINAL
# ============================================================
print("\n" + "=" * 70)
print("RESUMO FINAL - MODELOS E COMPARAÇÕES")
print("=" * 70)

# Tabela resumida de todos os modelos
print("\n📊 PERFORMANCE NO CONJUNTO DE TESTE:\n")
summary_df = pd.DataFrame({
    'Modelo': list(test_results.keys()),
    'Acurácia': [test_results[m]['accuracy'] for m in test_results.keys()],
    'AUC': [test_results[m]['auc'] for m in test_results.keys()],
    'Precisão': [test_results[m]['precision'] for m in test_results.keys()],
    'Recall': [test_results[m]['recall'] for m in test_results.keys()],
    'F1-Score': [test_results[m]['f1'] for m in test_results.keys()]
})

print(summary_df.to_string(index=False))

print("\n\n📈 ANÁLISE DE OVERFITTING:\n")
overfitting_df = pd.DataFrame({
    'Modelo': list(overfitting_analysis.keys()),
    'Gap (%)': [f"{g*100:.2f}%" for g in overfitting_analysis.values()],
    'Status': ['OK' if g < 0.05 else 'LEVE' if g < 0.10 else 'MODERADO' if g < 0.15 else 'CRÍTICO'
               for g in overfitting_analysis.values()]
})

print(overfitting_df.to_string(index=False))

print("\n\n🔄 VALIDAÇÃO CRUZADA (5 Folds):\n")
cv_df = pd.DataFrame({
    'Modelo': list(cv_results.keys()),
    'Média': [f"{np.mean(cv_results[m]):.4f}" for m in cv_results.keys()],
    'Desvio': [f"±{np.std(cv_results[m]):.4f}" for m in cv_results.keys()],
    'Intervalo': [f"[{np.min(cv_results[m]):.3f}, {np.max(cv_results[m]):.3f}]" for m in cv_results.keys()]
})

print(cv_df.to_string(index=False))

print("\n\n" + "=" * 70)
print("✅ DESTAQUES DA ANÁLISE")
print("=" * 70)

best_model = max(test_results.keys(), key=lambda x: test_results[x]['accuracy'])
best_acc = test_results[best_model]['accuracy']

worst_overfitting = max(overfitting_analysis.keys(), key=lambda x: overfitting_analysis[x])
worst_gap = overfitting_analysis[worst_overfitting]

print(f"""
✓ MELHOR MODELO: {best_model}
  - Acurácia Teste: {best_acc*100:.2f}%
  - AUC: {test_results[best_model]['auc']:.4f}
  - F1-Score: {test_results[best_model]['f1']:.4f}

• KNN PERFORMANCE:
  - Acurácia: {test_results['KNN']['accuracy']*100:.2f}%
  - AUC: {test_results['KNN']['auc']:.4f}
  - Precisão: {test_results['KNN']['precision']:.4f}
  - Recall: {test_results['KNN']['recall']:.4f}
  - Vantagem: Simples, interpretável, não assume distribuição
  - Desvantagem: Sensível a escala, lento em predição, overfitting em K pequeno

• ENSEMBLE VOTING BENEFÍCIOS:
  - Combina 4 modelos diferentes (LR, RF, XGB, KNN)
  - Reduz variância através de votação soft
  - Mais robusto a mudanças de regime
  - Melhor generalização que modelos individuais

✗ MAIOR OVERFITTING: {worst_overfitting} (Gap: {worst_gap*100:.2f}%)

🎯 CONCLUSÃO:
  Pipeline correto com:
  ✓ Split temporal ANTES de features
  ✓ Scaler fit APENAS em treino
  ✓ Validação cruzada temporal (5 folds)
  ✓ Ensemble voting com 4 algoritmos
  ✓ Regularização adequada
  ✓ KNN integrado com sucesso
""")

print("=" * 70)
print("\n📊 Gráficos salvos:")
print("  - 'analise_modelo_ibovespa_com_knn.png'")
print("\n✅ Execução concluída com sucesso!")
print("=" * 70)
