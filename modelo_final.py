# -*- coding: utf-8 -*-
"""
CÓDIGO CORRIGIDO - Solução para Overfitting e Vazamento de Dados
Adaptado para o arquivo Ibovespa.csv real
==================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
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
# 7. TREINAMENTO COM REGULARIZAÇÃO
# ============================================================
print("=" * 70)
print("TREINAMENTO DO MODELO XGBOOST")
print("=" * 70)

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

print("Parâmetros do modelo:")
print(f"  - max_depth: 4 (reduzido para evitar overfitting)")
print(f"  - subsample: 0.8 (regularização)")
print(f"  - colsample_bytree: 0.8 (regularização)")
print(f"  - reg_alpha: 0.1 (L1)")
print(f"  - reg_lambda: 1.0 (L2)\n")

xgb.fit(X_train_scaled, y_train, verbose=False)

print("Modelo treinado\n")

# ============================================================
# 8. AVALIAÇÃO NO CONJUNTO DE TREINO
# ============================================================
y_pred_train = xgb.predict(X_train_scaled)
y_pred_proba_train = xgb.predict_proba(X_train_scaled)[:, 1]

acc_train = accuracy_score(y_train, y_pred_train)
auc_train = roc_auc_score(y_train, y_pred_proba_train)

print("=" * 70)
print("RESULTADOS NO CONJUNTO DE TREINO")
print("=" * 70)
print(f"Acurácia:  {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"ROC-AUC:   {auc_train:.4f}\n")

# ============================================================
# 9. AVALIAÇÃO NO CONJUNTO DE TESTE
# ============================================================
y_pred_test = xgb.predict(X_test_scaled)
y_pred_proba_test = xgb.predict_proba(X_test_scaled)[:, 1]

acc_test = accuracy_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_pred_proba_test)

print("=" * 70)
print("RESULTADOS NO CONJUNTO DE TESTE")
print("=" * 70)
print(f"Acurácia:  {acc_test:.4f} ({acc_test*100:.2f}%)")
print(f"ROC-AUC:   {auc_test:.4f}\n")

# ============================================================
# 10. ANÁLISE DE OVERFITTING
# ============================================================
gap = acc_train - acc_test

print("=" * 70)
print("ANÁLISE DE OVERFITTING")
print("=" * 70)
print(f"Gap (Treino - Teste): {gap:.4f} ({gap*100:.2f}%)")

if gap > 0.10:
    status = "CRÍTICO - Modelo está memorizando dados"
elif gap > 0.05:
    status = "MODERADO - Possível overfitting"
else:
    status = "OK - Generalização adequada"

print(f"Status: {status}\n")

# ============================================================
# 11. CLASSIFICAÇÃO DETALHADA
# ============================================================
print("=" * 70)
print("CLASSIFICATION REPORT (TESTE)")
print("=" * 70)
print(classification_report(y_test, y_pred_test, target_names=['Tendência Baixa (0)', 'Tendência Alta (1)']))

# ============================================================
# 12. VALIDAÇÃO CRUZADA TEMPORAL
# ============================================================
print("=" * 70)
print("TIME SERIES CROSS-VALIDATION (5 folds)")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_scaled)):
    X_fold_train = X_train_scaled[train_idx]
    X_fold_test = X_train_scaled[test_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_test = y_train.iloc[test_idx]
    
    model_cv = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, use_label_encoder=False, eval_metric='logloss',
        verbosity=0
    )
    model_cv.fit(X_fold_train, y_fold_train)
    
    y_pred_fold = model_cv.predict(X_fold_test)
    acc_fold = accuracy_score(y_fold_test, y_pred_fold)
    cv_scores.append(acc_fold)
    
    print(f"Fold {fold+1}: {acc_fold:.4f} ({acc_fold*100:.2f}%)")

print(f"\nMédia CV:  {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
print(f"Intervalo: [{np.min(cv_scores):.4f}, {np.max(cv_scores):.4f}]")
print()

# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================
print("=" * 70)
print("IMPORTÂNCIA DAS FEATURES")
print("=" * 70)

feature_importance = pd.DataFrame({
    'Feature': X_cols,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.iterrows():
    barra = "█" * int(row['Importance'] * 100)
    print(f"{row['Feature']:20s}: {barra} {row['Importance']:.4f}")
print()

# ============================================================
# 14. TABELA DE RESULTADOS DOS ÚLTIMOS 30 DIAS
# ============================================================
print("=" * 70)
print("RESULTADOS DOS ÚLTIMOS 30 DIAS")
print("=" * 70)

resultados_table = pd.DataFrame({
    'Data': df_test['Data'].dt.strftime('%d/%m/%Y').values,
    'Preço': df_test['Ultimo'].values,
    'Real': y_test.values,
    'Predito': y_pred_test,
    'Acerto': (y_test.values == y_pred_test).astype(int)
})

resultados_table['Resultado'] = resultados_table['Acerto'].map({1: 'Sim', 0: 'Não'})

total_acertos = resultados_table['Acerto'].sum()
pct_acertos = (total_acertos / len(resultados_table)) * 100

print(resultados_table[['Data', 'Preço', 'Real', 'Predito', 'Resultado']].to_string(index=False))
print(f"\n{'='*70}")
print(f"TOTAL DE ACERTOS: {total_acertos}/{len(resultados_table)} ({pct_acertos:.2f}%)")
print(f"{'='*70}\n")

# ============================================================
# 15. VISUALIZAÇÕES
# ============================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# Gráfico 1: Matriz de Confusão
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred_test)
cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_percent, annot=cm, fmt='d', cmap='Blues', 
            xticklabels=['Baixa', 'Alta'], yticklabels=['Baixa', 'Alta'], ax=ax1,
            cbar_kws={'label': 'Percentual (%)'})
ax1.set_title(f"Matriz de Confusão\nAcurácia: {acc_test*100:.2f}%", fontweight='bold')
ax1.set_ylabel('Real')
ax1.set_xlabel('Predito')

# Gráfico 2: Real vs Predito
ax2 = fig.add_subplot(gs[0, 1])
dias = np.arange(len(y_test))
ax2.plot(dias, y_test.values, label="Real", marker="o", color="blue", markersize=7, linewidth=2)
ax2.plot(dias, y_pred_test, label="Predito", marker="x", color="red", markersize=8, 
         linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
ax2.set_title("Tendência Real vs Prevista (Últimos 30 dias)", fontweight='bold')
ax2.set_xlabel("Dias")
ax2.set_ylabel("Tendência")
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, len(y_test), 3))

# Gráfico 3: Treino vs Teste (Overfitting Detection)
ax3 = fig.add_subplot(gs[1, 0])
metrics_labels = ['Acurácia', 'AUC']
train_vals = [acc_train, auc_train]
test_vals = [acc_test, auc_test]
x = np.arange(len(metrics_labels))
width = 0.35

bars1 = ax3.bar(x - width/2, train_vals, width, label='Treino', color='#2E86AB', alpha=0.8)
bars2 = ax3.bar(x + width/2, test_vals, width, label='Teste', color='#A23B72', alpha=0.8)

ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Detecção de Overfitting\n(Treino vs Teste)', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics_labels)
ax3.legend()
ax3.set_ylim([0, 1])

for i, (v_train, v_test) in enumerate(zip(train_vals, test_vals)):
    ax3.text(i - width/2, v_train + 0.03, f'{v_train:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax3.text(i + width/2, v_test + 0.03, f'{v_test:.3f}', ha='center', fontsize=9, fontweight='bold')

# Gap indicator
gap_color = '#D62828' if gap > 0.05 else '#06A77D'
ax3.text(0.5, 0.15, f'Gap: {gap*100:.1f}%', transform=ax3.transAxes, 
         fontsize=12, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.3))

# Gráfico 4: Cross-Validation
ax4 = fig.add_subplot(gs[1, 1])
folds = np.arange(1, len(cv_scores) + 1)
ax4.plot(folds, cv_scores, marker='o', linewidth=2.5, markersize=8, color='#2E86AB', label='Fold Accuracy')
ax4.axhline(y=np.mean(cv_scores), color='blue', linestyle='--', linewidth=2, 
            label=f'Média CV: {np.mean(cv_scores):.3f}')
ax4.axhline(y=acc_test, color='green', linestyle='--', linewidth=2, 
            label=f'Teste: {acc_test:.3f}')
ax4.fill_between(folds, 
                 np.array(cv_scores) - np.std(cv_scores),
                 np.array(cv_scores) + np.std(cv_scores),
                 alpha=0.2, color='blue')
ax4.set_xlabel('CV Fold', fontweight='bold')
ax4.set_ylabel('Accuracy', fontweight='bold')
ax4.set_title('Time Series Cross-Validation', fontweight='bold')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(folds)

# Gráfico 5: Feature Importance
ax5 = fig.add_subplot(gs[2, :])
top_features = feature_importance.head(11)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
bars = ax5.barh(range(len(top_features)), top_features['Importance'].values, color=colors)
ax5.set_yticks(range(len(top_features)))
ax5.set_yticklabels(top_features['Feature'].values, fontsize=10)
ax5.set_xlabel('Importância', fontweight='bold')
ax5.set_title('Feature Importance (XGBoost)', fontweight='bold')
ax5.invert_yaxis()

for i, v in enumerate(top_features['Importance'].values):
    ax5.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

plt.suptitle('ANÁLISE COMPLETA DO MODELO IBOVESPA (CORRIGIDO)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig('analise_modelo_ibovespa_corrigido.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo: 'analise_modelo_ibovespa_corrigido.png'")
plt.show()

# ============================================================
# 16. RESUMO FINAL
# ============================================================
print("\n" + "=" * 70)
print("RESUMO FINAL")
print("=" * 70)
print(f"""
PIPELINE CORRIGIDO IMPLEMENTADO:
   1. Separação treino/teste ANTES de calcular indicadores
   2. Indicadores criados SEPARADAMENTE em cada set
   3. StandardScaler fit APENAS em dados de treino
   4. Regularização aumentada (L1, L2, subsample, colsample)
   5. Validação cruzada temporal (5 folds)

RESULTADOS:
   Treino Accuracy: {acc_train*100:.2f}%
   Teste  Accuracy: {acc_test*100:.2f}%
   Overfitting Gap: {gap*100:.2f}%
   
   Treino AUC: {auc_train:.4f}
   Teste  AUC: {auc_test:.4f}
   
   CV Média:   {np.mean(cv_scores)*100:.2f}% (±{np.std(cv_scores)*100:.2f}%)
   Acertos:    {total_acertos}/30 ({pct_acertos:.2f}%)

{'Modelo bem generalizado!' if gap < 0.05 else 'Considere ajustar hiperparâmetros'}
""")
print("=" * 70)
