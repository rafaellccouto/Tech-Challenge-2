"""
Gerador de Gráficos - Modelo de Previsão Ibovespa
Cria visualizações para apresentação:
- Série histórica do Ibovespa
- Previsto vs Real (últimos 30 dias)
- Matriz de Confusão
- Curva ROC
- Performance por diferentes tamanhos de dados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, 
                            accuracy_score, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("GERANDO GRAFICOS PARA APRESENTACAO")
print("=" * 80)

# ==== CARREGAR E PREPARAR DADOS ====
print("\n1. Carregando dados...")

df = pd.read_csv('Ibovespa.csv')
df.columns = df.columns.str.strip()
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df = df.sort_values('Data').reset_index(drop=True)

for col in ['Último', 'Abertura', 'Máxima', 'Mínima']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Vol.'] = df['Vol.'].astype(str).str.replace(',', '.').str.replace('B', '000000000').str.replace('M', '000000')
df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce')

df['Var%'] = df['Var%'].astype(str).str.strip().str.replace('%', '').str.replace(',', '.')
df['Var%'] = pd.to_numeric(df['Var%'], errors='coerce')

# Preparar features
def create_robust_features(data):
    df_f = data.copy()
    df_f['mom_1'] = df_f['Var%'].shift(1)
    df_f['mom_3'] = df_f['Var%'].shift(1) + df_f['Var%'].shift(2) + df_f['Var%'].shift(3)
    df_f['mom_5'] = df_f['Var%'].rolling(5, min_periods=1).sum().shift(1)
    gain_10 = (df_f['Var%'] > 0).rolling(10, min_periods=1).sum()
    df_f['strength_10'] = (gain_10 - 5) / 5
    df_f['vol_10'] = df_f['Var%'].rolling(10, min_periods=1).std()
    df_f['sma_20'] = df_f['Último'].rolling(20, min_periods=1).mean()
    df_f['above_sma'] = (df_f['Último'] > df_f['sma_20']).astype(int)
    df_f['range_pct'] = ((df_f['Máxima'] - df_f['Mínima']) / df_f['Mínima'] * 100).shift(1)
    return df_f

split_idx = len(df) - 30
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

df_train = create_robust_features(df_train)
df_test = create_robust_features(df_test)

# Target
df_train['target'] = (df_train['Var%'].shift(-1) > 0).astype(int)
df_test['target'] = (df_test['Var%'].shift(-1) > 0).astype(int)

df_train = df_train.dropna(subset=['target'])
df_test = df_test.dropna(subset=['target'])

feature_cols = ['mom_1', 'mom_3', 'mom_5', 'strength_10', 'vol_10', 'above_sma', 'range_pct']
df_train = df_train.dropna(subset=feature_cols)
df_test = df_test.dropna(subset=feature_cols)

X_train = df_train[feature_cols].fillna(0)
y_train = df_train['target']
X_test = df_test[feature_cols].fillna(0)
y_test = df_test['target']

# Normalizar
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Treinar modelo
print("2. Treinando modelo...")
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.6, random_state=42, verbosity=0))
    ],
    voting='soft',
    n_jobs=-1
)
ensemble.fit(X_train_s, y_train)

y_pred = ensemble.predict(X_test_s)
y_proba = ensemble.predict_proba(X_test_s)[:, 1]

print("\n✅ Modelo treinado!")

# ==== GRÁFICO 1: SÉRIE HISTÓRICA DO IBOVESPA ====
print("\n3. Gerando SERIE HISTORICA...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Preço
ax1.plot(df['Data'], df['Último'], linewidth=2, color='#1f77b4', label='Preço de Fechamento')
ax1.fill_between(df['Data'], df['Mínima'], df['Máxima'], alpha=0.2, color='#1f77b4', label='Range Dia')
ax1.axvline(df.iloc[split_idx]['Data'], color='red', linestyle='--', linewidth=2, label='Split Treino/Teste')
ax1.set_title('Série Histórica do Ibovespa (501 dias)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Preço (pontos)', fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Variação
colors = ['green' if x > 0 else 'red' for x in df['Var%']]
ax2.bar(df['Data'], df['Var%'], color=colors, alpha=0.6, width=1)
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(df.iloc[split_idx]['Data'], color='red', linestyle='--', linewidth=2, label='Split Treino/Teste')
ax2.set_title('Variação Diária (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Variação %', fontsize=11)
ax2.set_xlabel('Data', fontsize=11)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafico_01_serie_historica.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvo: grafico_01_serie_historica.png")
plt.close()

# ==== GRÁFICO 2: PREVISTO vs REAL (ÚLTIMOS 30 DIAS) ====
print("4. Gerando PREVISTO vs REAL...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Proccess data for visualization
test_dates = pd.to_datetime(df_test['Data'].values[:len(y_test)])
test_prices = df_test['Último'].values[:len(y_test)]
test_var = df_test['Var%'].values[:len(y_test)]

# Gráfico 1: Preço com previsões
colors_pred = ['green' if p == 1 else 'red' for p in y_pred]
colors_real = ['green' if r == 1 else 'red' for r in y_test.values]

x_pos = np.arange(len(test_dates))
width = 0.35

# Real
for i, (date, real, var) in enumerate(zip(test_dates, test_var, colors_real)):
    ax1.scatter(i, var, s=200, color=colors_real[i], marker='o', alpha=0.7, label='Real' if i == 0 else '', zorder=3)

# Predito
for i, (date, pred) in enumerate(zip(test_dates, y_pred)):
    offset = 0.15
    ax1.scatter(i + offset, y_proba[i] * 2 - 1, s=150, color=colors_pred[i], marker='s', alpha=0.5, label='Predito' if i == 0 else '', zorder=2)

ax1.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
ax1.set_title('Previsões vs Real (Últimos 30 Dias)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Variação (%)', fontsize=11)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([pd.Timestamp(d).strftime('%d/%m') for d in test_dates], rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(loc='best', fontsize=10)

# Gráfico 2: Probabilidade vs Real
for i, (date, prob, real) in enumerate(zip(test_dates, y_proba, y_test.values)):
    color = 'green' if real == 1 else 'red'
    marker = 'o' if prob > 0.5 else 'x'
    ax2.scatter(i, prob, s=200, color=color, marker=marker, alpha=0.7)

ax2.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (50%)')
ax2.set_title('Probabilidade de Alta Predita', fontsize=14, fontweight='bold')
ax2.set_ylabel('Probabilidade', fontsize=11)
ax2.set_xlabel('Data', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([pd.Timestamp(d).strftime('%d/%m') for d in test_dates], rotation=45, ha='right')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('grafico_02_previsto_vs_real.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvo: grafico_02_previsto_vs_real.png")
plt.close()

# ==== GRÁFICO 3: MATRIZ DE CONFUSÃO ====
print("5. Gerando MATRIZ DE CONFUSAO...")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Baixa', 'Alta'],
            yticklabels=['Baixa', 'Alta'],
            annot_kws={'size': 16, 'weight': 'bold'},
            cbar_kws={'label': 'Contagem'})

ax.set_title('Matriz de Confusão - Conjunto de Teste', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Real', fontsize=12, fontweight='bold')
ax.set_xlabel('Predito', fontsize=12, fontweight='bold')

# Adicionar métricas
accuracy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

textstr = f'Acurácia: {accuracy:.1%}\nSensibilidade: {sensitivity:.1%}\nEspecificidade: {specificity:.1%}'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('grafico_03_matriz_confusao.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvo: grafico_03_matriz_confusao.png")
plt.close()

# ==== GRÁFICO 4: CURVA ROC ====
print("6. Gerando CURVA ROC...")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(10, 8))

# Curva ROC
ax.plot(fpr, tpr, color='#1f77b4', linewidth=2.5, 
        label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, 
        label='Random Classifier (AUC = 0.500)')

# Highlight ponto de operação atual
current_idx = np.argmin(np.abs(thresholds - 0.5))
ax.scatter(fpr[current_idx], tpr[current_idx], color='green', s=200, 
          marker='o', zorder=5, label=f'Threshold=0.5')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12, fontweight='bold')
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12, fontweight='bold')
ax.set_title('Curva ROC - Desempenho do Modelo', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafico_04_curva_roc.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvo: grafico_04_curva_roc.png")
plt.close()

# ==== GRÁFICO 5: PERFORMANCE POR TAMANHO DE DADOS (VALIDAÇÃO CRUZADA) ====
print("7. Gerando PERFORMANCE vs TAMANHO DE DADOS...")

tscv = TimeSeriesSplit(n_splits=5)
fold_results = []
fold_labels = []
train_sizes = []
test_accuracies = []
train_accuracies = []

for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X_train_s)):
    X_cv_train = X_train_s[train_idx]
    y_cv_train = y_train.iloc[train_idx]
    X_cv_test = X_train_s[test_idx]
    y_cv_test = y_train.iloc[test_idx]
    
    train_size = len(X_cv_train)
    test_size = len(X_cv_test)
    
    model_cv = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.6, random_state=42, verbosity=0))
        ],
        voting='soft',
        n_jobs=-1
    )
    
    model_cv.fit(X_cv_train, y_cv_train)
    
    train_acc = accuracy_score(y_cv_train, model_cv.predict(X_cv_train))
    test_acc = accuracy_score(y_cv_test, model_cv.predict(X_cv_test))
    
    train_sizes.append(train_size)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    fold_labels.append(f'Fold {fold_id+1}')

# Gráfico
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Acurácia por Fold
x_pos = np.arange(len(fold_labels))
width = 0.35

ax1.bar(x_pos - width/2, train_accuracies, width, label='Acurácia Treino', color='#2ca02c', alpha=0.8)
ax1.bar(x_pos + width/2, test_accuracies, width, label='Acurácia Teste', color='#d62728', alpha=0.8)

ax1.axhline(np.mean(test_accuracies), color='#d62728', linestyle='--', linewidth=2, 
           label=f'Média Teste ({np.mean(test_accuracies):.1%})')
ax1.set_title('Performance por Fold (Validação Cruzada Temporal)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Acurácia', fontsize=11)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(fold_labels)
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(fontsize=10)

# Gráfico 2: Tamanho de Treino vs Performance
ax2.plot(train_sizes, train_accuracies, marker='o', linewidth=2.5, markersize=8, 
        label='Acurácia Treino', color='#2ca02c')
ax2.plot(train_sizes, test_accuracies, marker='s', linewidth=2.5, markersize=8, 
        label='Acurácia Teste', color='#d62728')

ax2.fill_between(train_sizes, train_accuracies, test_accuracies, alpha=0.2, color='gray',
                label='Gap Treino-Teste')

ax2.set_title('Performance vs Tamanho de Dados Treino', fontsize=13, fontweight='bold')
ax2.set_xlabel('Tamanho do Conjunto de Treino', fontsize=11)
ax2.set_ylabel('Acurácia', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('grafico_05_performance_vs_tamanho.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvo: grafico_05_performance_vs_tamanho.png")
plt.close()

# ==== GRÁFICO 6: DISTRIBUIÇÃO DE PROBABILIDADES ====
print("8. Gerando DISTRIBUICAO DE PROBABILIDADES...")

fig, ax = plt.subplots(figsize=(12, 7))

# Histograma separado por classe real
proba_altas = y_proba[y_test.values == 1]
proba_baixas = y_proba[y_test.values == 0]

ax.hist(proba_baixas, bins=10, color='red', alpha=0.6, label=f'Real Baixa (n={len(proba_baixas)})', edgecolor='black')
ax.hist(proba_altas, bins=10, color='green', alpha=0.6, label=f'Real Alta (n={len(proba_altas)})', edgecolor='black')

ax.axvline(0.5, color='black', linestyle='--', linewidth=2.5, label='Threshold (50%)')
ax.set_title('Distribuição de Probabilidades Preditas', fontsize=14, fontweight='bold')
ax.set_xlabel('Probabilidade de Alta Predita', fontsize=11)
ax.set_ylabel('Frequência', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('grafico_06_distribuicao_probabilidades.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvo: grafico_06_distribuicao_probabilidades.png")
plt.close()

# ==== GRÁFICO 7: FEATURE IMPORTANCE ====
print("9. Gerando FEATURE IMPORTANCE...")

rf_model = ensemble.estimators_[1]
importances = rf_model.feature_importances_
features_sorted = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(12, 7))

features, imps = zip(*features_sorted)
colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))

bars = ax.barh(features, imps, color=colors_imp, edgecolor='black', linewidth=1.5)

# Adicionar valores nas barras
for i, (bar, imp) in enumerate(zip(bars, imps)):
    ax.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=10, fontweight='bold')

ax.set_title('Importância das Features (Random Forest)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importância', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('grafico_07_feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvo: grafico_07_feature_importance.png")
plt.close()

# ==== GRÁFICO 8: COMPARAÇÃO TREINO vs TESTE ====
print("10. Gerando TREINO vs TESTE...")

y_train_pred = ensemble.predict(X_train_s)
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 7))

datasets = ['Treino\n(471 amostras)', 'Teste\n(27 amostras)']
accuracies = [acc_train, acc_test]
colors_bar = ['#2ca02c', '#d62728']

bars = ax.bar(datasets, accuracies, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2, width=0.6)

# Adicionar valores
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{acc:.1%}',
           ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylim([0, 1])
ax.set_ylabel('Acurácia', fontsize=12, fontweight='bold')
ax.set_title('Acurácia: Treino vs Teste (Análise de Overfitting)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Adicionar gap
gap = acc_train - acc_test
ax.annotate('', xy=(0, acc_test), xytext=(0, acc_train),
           arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0.15, (acc_train + acc_test)/2, f'Gap\n{gap:.1%}', fontsize=11, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('grafico_08_treino_vs_teste.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvo: grafico_08_treino_vs_teste.png")
plt.close()

# ==== RESUMO FINAL ====
print("\n" + "=" * 80)
print("TODOS OS GRAFICOS GERADOS COM SUCESSO!")
print("=" * 80)

print("""
GRÁFICOS CRIADOS:

1. grafico_01_serie_historica.png
   └─ Série temporal do Ibovespa + variações diárias + split treino/teste

2. grafico_02_previsto_vs_real.png
   └─ Comparação de previsões vs valores reais (últimos 30 dias)

3. grafico_03_matriz_confusao.png
   └─ Matriz de confusão com métricas (acurácia, sensibilidade, especificidade)

4. grafico_04_curva_roc.png
   └─ Curva ROC com AUC score

5. grafico_05_performance_vs_tamanho.png
   └─ Performance por fold de validação cruzada + aprendizado com tamanho

6. grafico_06_distribuicao_probabilidades.png
   └─ Distribuição de probabilidades preditas separada por classe real

7. grafico_07_feature_importance.png
   └─ Importância das 7 features do modelo

8. grafico_08_treino_vs_teste.png
   └─ Análise de overfitting (comparação treino/teste)

USO PARA APRESENTAÇÃO:
   - Todos salvos em alta resolução (300 DPI)
   - Formato PNG (compatível com PowerPoint/Google Slides)
   - Dimensões otimizadas para projeção (14" x 8")

DICAS:
   - Usar grafico_02 para mostrar previsões práticas
   - Usar grafico_03 para explicar erros do modelo
   - Usar grafico_04 para discussão técnica (ROC/AUC)
   - Usar grafico_05 para mostrar consistência (CV)
   - Usar grafico_06 para análise de decisões
   - Usar grafico_07 para feature engineering
   - Usar grafico_08 para explicar que não há overfitting crítico
""")

print("\n" + "=" * 80)
