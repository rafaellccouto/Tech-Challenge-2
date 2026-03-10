"""
VISUALIZAÇÕES PARA APRESENTAÇÃO - Tech Challenge 2
Gráficos profissionais prontos para apresentação ao público
- Curva ROC
- AUC comparativo
- Matriz de Confusão
- Série Histórica vs Previsto
- Performance Ensemble vs Individuais
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, auc,
                            accuracy_score, precision_score, recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Estilo para apresentação
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Cores para consistência
CORES = {
    'logistic': '#1f77b4',    # Azul
    'rf': '#ff7f0e',          # Laranja
    'xgb': '#2ca02c',         # Verde
    'knn': '#d62728',         # Vermelho
    'ensemble': '#9467bd'     # Roxo (destaque)
}

print("=" * 80)
print("VISUALIZAÇÕES PARA APRESENTAÇÃO - Tech Challenge 2")
print("=" * 80)

# ============================================================
# CARREGAR E PREPARAR DADOS
# ============================================================

print("\n[1/5] Carregando dados...")
df = pd.read_csv('Ibovespa.csv')
print(f"✓ {len(df)} dias carregados")

# Renomear colunas
df = df.rename(columns={'Último': 'Ultimo', 'Abertura': 'Abertura', 'Máxima': 'Maxima', 'Mínima': 'Minima', 'Vol.': 'Vol', 'Var%': 'VarPerc'})

# Converter Data para datetime
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')

# Processar colunas de preço (remover separadores)
for col in ['Ultimo', 'Abertura', 'Maxima', 'Minima']:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Processar Volume (remover 'B')
df['Vol'] = df['Vol'].astype(str).str.replace('B', '', regex=False)
df['Vol'] = df['Vol'].str.replace(',', '.', regex=False)
df['Vol'] = pd.to_numeric(df['Vol'], errors='coerce')

# Processar Variação percentual
df['VarPerc'] = df['VarPerc'].astype(str).str.replace('%', '', regex=False)
df['VarPerc'] = df['VarPerc'].str.replace(',', '.', regex=False)
df['VarPerc'] = pd.to_numeric(df['VarPerc'], errors='coerce')

# Criar target: 1 se próximo preço > preço atual (SOBE), 0 senão (DESCE)
df['Tendencia'] = (df['Ultimo'].shift(-1) > df['Ultimo']).astype(int)

# Remover linhas sem dados válidos
df = df.dropna()

print(f"✓ {len(df)} registros válidos após processamento")

# Usar últimas 44 dias para teste (Nov-Dez)
split_point = len(df) - 44

df_train = df.iloc[:split_point].copy()
df_test = df.iloc[split_point:].copy()

print(f"✓ Split temporal: {len(df_train)} treino, {len(df_test)} teste")

# Preparar features
features_list = ['Abertura', 'Maxima', 'Minima', 'Vol', 'VarPerc']
X_train = df_train[features_list].values
X_test = df_test[features_list].values
y_train = df_train['Tendencia'].values
y_test = df_test['Tendencia'].values

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# TREINAR MODELOS
# ============================================================

print("\n[2/5] Treinando modelos...")

# Logistic Regression
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
lr.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=5,
                           min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# XGBoost
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=1.0,
                   random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
xgb.fit(X_train_scaled, y_train)

# KNN K=10 (Otimizado)
knn = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', n_jobs=-1)
knn.fit(X_train_scaled, y_train)

# Ensemble Voting
voting_clf = VotingClassifier(
    estimators=[
        ('logistic', LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=5,
                                     min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=1.0,
                             random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)),
        ('knn', KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', n_jobs=-1))
    ],
    voting='soft',
    weights=[1, 1.2, 1.5, 0.8]
)
voting_clf.fit(X_train_scaled, y_train)

print("✓ 5 modelos treinados com sucesso")

# ============================================================
# GRÁFICO 1: ROC CURVES COMPARATIVAS (Todos os Modelos)
# ============================================================

print("\n[3/5] Gerando gráficos...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ROC Curves - Comparação de Modelos', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Logistic Regression
ax = axes[0, 0]
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
auc_lr = roc_auc_score(y_test, y_proba_lr)
ax.plot(fpr_lr, tpr_lr, color=CORES['logistic'], lw=3, label=f'AUC = {auc_lr:.3f}')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, alpha=0.5, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Logistic Regression')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Plot 2: Random Forest
ax = axes[0, 1]
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)
ax.plot(fpr_rf, tpr_rf, color=CORES['rf'], lw=3, label=f'AUC = {auc_rf:.3f}')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Random Forest')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Plot 3: XGBoost
ax = axes[0, 2]
y_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)
ax.plot(fpr_xgb, tpr_xgb, color=CORES['xgb'], lw=3, label=f'AUC = {auc_xgb:.3f}')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('XGBoost')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Plot 4: KNN K=10
ax = axes[1, 0]
y_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
auc_knn = roc_auc_score(y_test, y_proba_knn)
ax.plot(fpr_knn, tpr_knn, color=CORES['knn'], lw=3, label=f'AUC = {auc_knn:.3f}')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('KNN (K=10)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Plot 5: Ensemble Voting (DESTAQUE)
ax = axes[1, 1]
y_proba_ensemble = voting_clf.predict_proba(X_test_scaled)[:, 1]
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_proba_ensemble)
auc_ensemble = roc_auc_score(y_test, y_proba_ensemble)
ax.plot(fpr_ensemble, tpr_ensemble, color=CORES['ensemble'], lw=4, label=f'AUC = {auc_ensemble:.3f}')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Ensemble Voting (FINAL)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
ax.set_facecolor('#f0f0f0')  # Destaque visual

# Plot 6: Comparação AUC
ax = axes[1, 2]
models = ['Logistic', 'RF', 'XGBoost', 'KNN', 'Ensemble']
aucs = [auc_lr, auc_rf, auc_xgb, auc_knn, auc_ensemble]
colors = [CORES['logistic'], CORES['rf'], CORES['xgb'], CORES['knn'], CORES['ensemble']]
bars = ax.bar(models, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('AUC Score', fontweight='bold')
ax.set_title('Comparação AUC (Destaque: Ensemble)')
ax.set_ylim([0.5, 0.85])
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, auc in zip(bars, aucs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('apresentacao_01_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 salvo: apresentacao_01_roc_curves.png")
plt.close()

# ============================================================
# GRÁFICO 2: MATRIZES DE CONFUSÃO
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Matrizes de Confusão - Comparação de Modelos', fontsize=16, fontweight='bold', y=0.995)

models_to_plot = [
    ('Logistic Regression', lr, CORES['logistic']),
    ('Random Forest', rf, CORES['rf']),
    ('XGBoost', xgb, CORES['xgb']),
    ('KNN (K=10)', knn, CORES['knn']),
    ('Ensemble Voting', voting_clf, CORES['ensemble']),
]

for idx, (name, model, color) in enumerate(models_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotar matriz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
               xticklabels=['Desce (0)', 'Sobe (1)'],
               yticklabels=['Desce (0)', 'Sobe (1)'],
               annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    # Calcular métricas
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    ax.set_title(f'{name}\nAcc: {acc:.1%} | Prec: {prec:.1%} | Rec: {rec:.1%}',
                fontweight='bold', fontsize=12)
    ax.set_ylabel('Real')
    ax.set_xlabel('Previsto')

# Remover gráfico vazio
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('apresentacao_02_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 salvo: apresentacao_02_confusion_matrices.png")
plt.close()

# ============================================================
# GRÁFICO 3: SÉRIE HISTÓRICA vs PREVISTO
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Série Histórica: Real vs Previsto', fontsize=16, fontweight='bold', y=0.995)

# Obter índices de teste (últimos 44 dias)
y_pred_ensemble = voting_clf.predict(X_test_scaled)

# Plot 1: Série histórica completa (teste)
ax = axes[0, 0]
ax.plot(range(len(y_test)), y_test, 'o-', label='Real', color='black', linewidth=2, markersize=6)
ax.plot(range(len(y_pred_ensemble)), y_pred_ensemble, 's--', label='Previsto (Ensemble)', 
       color=CORES['ensemble'], linewidth=2, markersize=6, alpha=0.8)
ax.set_xlabel('Dia de Teste')
ax.set_ylabel('Movimento')
ax.set_title('Série Temporal: Real vs Previsto')
ax.set_yticks([0, 1])
ax.set_yticklabels(['DESCE', 'SOBE'])
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Zoom últimos 20 dias
ax = axes[0, 1]
zoom_range = min(20, len(y_test))
ax.plot(range(zoom_range), y_test[-zoom_range:], 'o-', label='Real', 
       color='black', linewidth=2, markersize=8)
ax.plot(range(zoom_range), y_pred_ensemble[-zoom_range:], 's--', label='Previsto', 
       color=CORES['ensemble'], linewidth=2, markersize=8, alpha=0.8)
ax.set_xlabel('Dia de Teste')
ax.set_ylabel('Movimento')
ax.set_title(f'Zoom: Últimos {zoom_range} Dias')
ax.set_yticks([0, 1])
ax.set_yticklabels(['DESCE', 'SOBE'])
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Probabilidades do Ensemble
ax = axes[1, 0]
ax.fill_between(range(len(y_proba_ensemble)), 0.5, y_proba_ensemble, 
                where=(y_proba_ensemble >= 0.5), alpha=0.3, color='green', label='Prevê SOBE')
ax.fill_between(range(len(y_proba_ensemble)), 0.5, y_proba_ensemble, 
                where=(y_proba_ensemble < 0.5), alpha=0.3, color='red', label='Prevê DESCE')
ax.plot(y_proba_ensemble, 'o-', color=CORES['ensemble'], linewidth=2, markersize=5)
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Dia de Teste')
ax.set_ylabel('Probabilidade')
ax.set_title('Confiança do Ensemble por Dia')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Acertos vs Erros
ax = axes[1, 1]
acertos = (y_pred_ensemble == y_test).astype(int)
ax.bar(range(len(acertos)), acertos, color=['red' if x == 0 else 'green' for x in acertos], 
      alpha=0.7, edgecolor='black', linewidth=1)
ax.set_xlabel('Dia de Teste')
ax.set_ylabel('Acerto?')
ax.set_title('Acertos (Verde) vs Erros (Vermelho)')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Erro', 'Acerto'])
ax.grid(axis='y', alpha=0.3)

# Adicionar taxa de acerto
acc_rate = acertos.mean()
ax.text(0.98, 0.95, f'Taxa de Acerto: {acc_rate:.1%}', 
       transform=ax.transAxes, fontsize=12, fontweight='bold',
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('apresentacao_03_serie_historica.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 3 salvo: apresentacao_03_serie_historica.png")
plt.close()

# ============================================================
# GRÁFICO 4: PERFORMANCE METRICS COMPARATIVO
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Métricas de Performance - Comparação de Modelos', fontsize=16, fontweight='bold', y=0.995)

# Calcular métricas para todos os modelos
results = {}
for name, model, color in models_to_plot:
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'color': color
    }

# Plot 1: Accuracy
ax = axes[0, 0]
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
colors = [results[n]['color'] for n in names]
bars = ax.bar(names, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Acurácia')
ax.set_ylim([0.6, 0.85])
ax.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.1%}',
           ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.tick_params(axis='x', rotation=45)

# Plot 2: AUC-ROC
ax = axes[0, 1]
aucs = [results[n]['auc'] for n in names]
bars = ax.bar(names, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('AUC-ROC', fontweight='bold')
ax.set_title('Área Sob a Curva ROC')
ax.set_ylim([0.65, 0.85])
ax.grid(axis='y', alpha=0.3)
for bar, auc_val in zip(bars, aucs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{auc_val:.3f}',
           ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.tick_params(axis='x', rotation=45)

# Plot 3: Precision vs Recall
ax = axes[1, 0]
precs = [results[n]['precision'] for n in names]
recs = [results[n]['recall'] for n in names]
x = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x - width/2, precs, width, label='Precision', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, recs, width, label='Recall', alpha=0.7, edgecolor='black')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Precision vs Recall')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylim([0.6, 1.0])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: F1-Score
ax = axes[1, 1]
f1s = [results[n]['f1'] for n in names]
bars = ax.bar(names, f1s, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('F1-Score', fontweight='bold')
ax.set_title('F1-Score (Harmônico entre Precision e Recall)')
ax.set_ylim([0.6, 1.0])
ax.grid(axis='y', alpha=0.3)
for bar, f1 in zip(bars, f1s):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{f1:.3f}',
           ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('apresentacao_04_performance_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 4 salvo: apresentacao_04_performance_metrics.png")
plt.close()

# ============================================================
# GRÁFICO 5: PROBABILIDADES DOS 5 MODELOS (Stacked)
# ============================================================

fig, ax = plt.subplots(figsize=(16, 8))

y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
y_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
y_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]

x_range = range(len(y_test))

# Plotar cada modelo
ax.plot(x_range, y_proba_lr, 'o-', label='Logistic', color=CORES['logistic'], linewidth=2, markersize=4)
ax.plot(x_range, y_proba_rf, 's-', label='Random Forest', color=CORES['rf'], linewidth=2, markersize=4)
ax.plot(x_range, y_proba_xgb, '^-', label='XGBoost', color=CORES['xgb'], linewidth=2, markersize=4)
ax.plot(x_range, y_proba_knn, 'd-', label='KNN (K=10)', color=CORES['knn'], linewidth=2, markersize=4)
ax.plot(x_range, y_proba_ensemble, 'o-', label='Ensemble (FINAL)', color=CORES['ensemble'], 
       linewidth=3, markersize=6, alpha=0.9)

# Threshold
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (0.5)')

# Preencher real
ax.fill_between(x_range, 0, 1, where=(y_test == 1), alpha=0.1, color='green', label='Real: SOBE')
ax.fill_between(x_range, 0, 1, where=(y_test == 0), alpha=0.1, color='red', label='Real: DESCE')

ax.set_xlabel('Dia de Teste', fontweight='bold', fontsize=12)
ax.set_ylabel('Probabilidade de SOBE', fontweight='bold', fontsize=12)
ax.set_title('Probabilidades Preditas - Todos os Modelos + Ensemble', fontweight='bold', fontsize=14)
ax.set_ylim([0, 1])
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('apresentacao_05_probabilidades.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 5 salvo: apresentacao_05_probabilidades.png")
plt.close()

# ============================================================
# RESUMO DE RESULTADOS
# ============================================================

print("\n" + "=" * 80)
print("RESUMO DE RESULTADOS - ENSEMBLE v2.1")
print("=" * 80)

y_pred_ens = voting_clf.predict(X_test_scaled)
cm_ens = confusion_matrix(y_test, y_pred_ens)
tn, fp, fn, tp = cm_ens.ravel()

print(f"\n✓ Acurácia Ensemble:      {accuracy_score(y_test, y_pred_ens):.1%}")
print(f"✓ AUC-ROC Ensemble:       {auc_ensemble:.4f}")
print(f"✓ Precision:              {precision_score(y_test, y_pred_ens):.1%}")
print(f"✓ Recall (Sensibilidade): {recall_score(y_test, y_pred_ens):.1%}")
print(f"✓ F1-Score:               {f1_score(y_test, y_pred_ens):.4f}")

print(f"\nMatriz de Confusão Ensemble:")
print(f"  Verdadeiros Negativos (TN):  {tn}")
print(f"  Falsos Positivos (FP):       {fp}")
print(f"  Falsos Negativos (FN):       {fn}")
print(f"  Verdadeiros Positivos (TP):  {tp}")
print(f"  Total de Acertos:            {tp + tn} / {len(y_test)}")

print(f"\n{'Model':<20} {'Accuracy':<12} {'AUC':<10} {'Precision':<12} {'Recall':<10}")
print("-" * 70)
for name, model, color in models_to_plot:
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"{name:<20} {acc:>10.1%}   {auc_val:>8.4f}  {prec:>10.1%}   {rec:>8.1%}")

print("\n" + "=" * 80)
print("GRÁFICOS GERADOS COM SUCESSO!")
print("=" * 80)
print("\nArquivos criados para apresentação:")
print("  ✓ apresentacao_01_roc_curves.png")
print("  ✓ apresentacao_02_confusion_matrices.png")
print("  ✓ apresentacao_03_serie_historica.png")
print("  ✓ apresentacao_04_performance_metrics.png")
print("  ✓ apresentacao_05_probabilidades.png")
print("\nPronto para apresentação profissional! 🎯")
