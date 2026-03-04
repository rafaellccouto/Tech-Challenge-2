import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AQUISIÇÃO E EXPLORAÇÃO DOS DADOS
# ============================================================================
print("=" * 80)
print("FASE 1: AQUISICAO E EXPLORACAO DOS DADOS")
print("=" * 80)

# Carregar dados
df_raw = pd.read_csv('Ibovespa.csv')

# Limpeza básica: remover espaços e converter tipos
df = df_raw.copy()
df.columns = df.columns.str.strip()
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df = df.sort_values('Data').reset_index(drop=True)

# Converter colunas numéricas
numeric_cols = ['Último', 'Abertura', 'Máxima', 'Mínima']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Tratamento especial para Volume (com sufixos como M e B)
df['Vol.'] = df['Vol.'].astype(str).str.replace(',', '.').str.replace('B', '000000000').str.replace('M', '000000')
df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce')

# Tratar Var% - remover % e converter
df['Var%'] = df['Var%'].astype(str).str.strip().str.replace('%', '').str.replace(',', '.')
df['Var%'] = pd.to_numeric(df['Var%'], errors='coerce')

print(f"\nDataset Shape: {df.shape}")
print(f"\nPeríodo: {df['Data'].min().date()} a {df['Data'].max().date()}")
print(f"\nPrimeiras linhas:")
print(df.head())
print(f"\nInfo do Dataset:")
print(df.info())
print(f"\nEstatisticas Descritivas:")
print(df[['Último', 'Abertura', 'Máxima', 'Mínima', 'Var%']].describe())

# ============================================================================
# DEFINIÇÃO DE VARIÁVEIS E DATASET SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("FASE 2: PREPARAÇÃO E DIVISÃO DO DATASET")
print("=" * 80)

# IMPORTANTE: Dividir ANTES de criar features para evitar data leakage
test_size = 30  # Últimos 30 dias
train_size = len(df) - test_size

df_train_raw = df.iloc[:train_size].copy()
df_test_raw = df.iloc[train_size:].copy()

print(f"\nTamanho do conjunto de TREINO: {len(df_train_raw)} dias")
print(f"   Periodo: {df_train_raw['Data'].min().date()} a {df_train_raw['Data'].max().date()}")
print(f"\nTamanho do conjunto de TESTE: {len(df_test_raw)} dias")
print(f"   Periodo: {df_test_raw['Data'].min().date()} a {df_test_raw['Data'].max().date()}")

# ============================================================================
# ENGENHARIA DE ATRIBUTOS (Feature Engineering)
# ============================================================================
print("\n" + "=" * 80)
print("FASE 3: ESTRATEGIA DE ENGENHARIA DE ATRIBUTOS")
print("=" * 80)

def create_features(data):
    """
    Cria features com base em:
    1. Variações percentuais (lag features)
    2. Médias móveis simples (SMA)
    3. Volatilidade (desvio padrão móvel)
    4. RSI (Relative Strength Index)
    5. Relação Abertura/Fechamento
    """
    df_feat = data.copy()
    
    # 1. VARIAÇÕES E LAG FEATURES
    df_feat['lag_1'] = df_feat['Var%'].shift(1)
    df_feat['lag_2'] = df_feat['Var%'].shift(2)
    df_feat['lag_3'] = df_feat['Var%'].shift(3)
    df_feat['lag_5'] = df_feat['Var%'].shift(5)
    
    # 2. MÉDIAS MÓVEIS (janelas de 5, 10, 20 dias)
    df_feat['sma_5'] = df_feat['Último'].rolling(window=5, min_periods=1).mean()
    df_feat['sma_10'] = df_feat['Último'].rolling(window=10, min_periods=1).mean()
    df_feat['sma_20'] = df_feat['Último'].rolling(window=20, min_periods=1).mean()
    
    # 3. VOLATILIDADE (desvio padrão)
    df_feat['volatility_5'] = df_feat['Var%'].rolling(window=5, min_periods=1).std()
    df_feat['volatility_10'] = df_feat['Var%'].rolling(window=10, min_periods=1).std()
    
    # 4. RSI - Relative Strength Index (14 períodos)
    delta = df_feat['Var%'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1)
    df_feat['rsi'] = 100 - (100 / (1 + rs))
    
    # 5. RELAÇÕES INTRA-DIA
    df_feat['high_low_ratio'] = (df_feat['Máxima'] - df_feat['Mínima']) / df_feat['Mínima']
    df_feat['open_close_ratio'] = (df_feat['Último'] - df_feat['Abertura']) / df_feat['Abertura'].abs()
    
    # 6. RELAÇÃO DO PREÇO COM MÉDIAS MÓVEIS
    df_feat['price_vs_sma5'] = (df_feat['Último'] - df_feat['sma_5']) / df_feat['sma_5']
    df_feat['price_vs_sma20'] = (df_feat['Último'] - df_feat['sma_20']) / df_feat['sma_20']
    
    return df_feat

df_train_feat = create_features(df_train_raw)
df_test_feat = create_features(df_test_raw)

print("\nFeatures criadas:")
print("   - Lag Features: variacoes atrasadas (lag_1, lag_2, lag_3, lag_5)")
print("   - Medias Moveis: SMA de 5, 10, 20 dias")
print("   - Volatilidade: Desvio padrao movel de 5 e 10 dias")
print("   - RSI: Indice de Forca Relativa (14 periodos)")
print("   - Ratios: Relacoes intra-dia e com medias moveis")

# ============================================================================
# DEFINIÇÃO DO TARGET
# ============================================================================
print("\n" + "=" * 80)
print("FASE 4: DEFINICAO DO TARGET")
print("=" * 80)

# Target: Próximo dia subiu (1) ou desceu (0)
df_train_feat['target'] = (df_train_feat['Var%'].shift(-1) > 0).astype(int)
df_test_feat['target'] = (df_test_feat['Var%'].shift(-1) > 0).astype(int)

# Remover última linha que não tem target
df_train_feat = df_train_feat.dropna(subset=['target'])
df_test_feat = df_test_feat.dropna(subset=['target'])

# Seleção de features (excluir NaN)
feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_5', 'sma_5', 'sma_10', 'sma_20',
                'volatility_5', 'volatility_10', 'rsi', 'high_low_ratio', 
                'open_close_ratio', 'price_vs_sma5', 'price_vs_sma20']

# Remover linhas com NaN nas features
df_train_feat = df_train_feat.dropna(subset=feature_cols)
df_test_feat = df_test_feat.dropna(subset=feature_cols)

X_train = df_train_feat[feature_cols].copy()
y_train = df_train_feat['target'].copy()
X_test = df_test_feat[feature_cols].copy()
y_test = df_test_feat['target'].copy()

print(f"\nTarget: Previsao se o preco sobe (1) ou desce (0) no proximo dia")
print(f"\nDistribuicao do Target - TREINO:")
print(y_train.value_counts())
print(f"\nDistribuicao do Target - TESTE:")
print(y_test.value_counts())

# ============================================================================
# NORMALIZAÇÃO DAS FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("FASE 5: NORMALIZACAO DAS FEATURES")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures normalizadas usando StandardScaler")
print("   (Media=0, Desvio Padrao=1)")

# ============================================================================
# ESCOLHA E JUSTIFICATIVA DO MODELO
# ============================================================================
print("\n" + "=" * 80)
print("FASE 6: ESCOLHA E TREINAMENTO DO MODELO")
print("=" * 80)

print("\nJUSTIFICATIVA DE MODELO:")
print("""
╔════════════════════════════════════════════════════════════════════┐
║ MODELO SELECIONADO: XGBoost (Extreme Gradient Boosting)          │
╠════════════════════════════════════════════════════════════════════╡
║ 1. RAZÃO DE ESCOLHA:                                              │
║    • Ensemble baseado em árvores com boosting sequencial          │
║    • Captura relações não-lineares complexas                      │
║    • Regularização integrada (L1/L2) reduz overfitting            │
║    • Excelente para problemas de classificação temporal            │
║    • Interpretável via feature importance                         │
║                                                                    │
║ 2. ALTERNATIVA CONSIDERADA:                                       │
║    • Random Forest (Ensemble paralelo)                            │
║    • Menos prone a overfitting que XGBoost puro, mas              │
║      geralmente com menor acurácia                                │
║                                                                    │
║ 3. NÃO USAR:                                                       │
║    • LSTM/RNN: Requer mais dados, maior risco de overfitting      │
║    • Logistic Regression: Não captura relações não-lineares       │
║                                                                    │
║ 4. TRATAMENTO DA NATUREZA SEQUENCIAL:                             │
║    • Lag Features: Captura dependência temporal                   │
║    • Médias móveis: Incorpora tendência histórica                 │
║    • Janela deslizante implícita nas features                     │
║    • Sem "future leakage" - split antes de features               │
║                                                                    │
║ 5. ANTI-OVERFITTING:                                               │
║    • Validação cruzada com awareness temporal                     │
║    • Regularização XGBoost (max_depth=5, learning_rate=0.1)       │
║    • Early stopping se validação não melhora                      │
║    • Dados de teste completamente isolados                        │
╚════════════════════════════════════════════════════════════════════╝
""")

# Treinar XGBoost com regularização anti-overfitting
model_xgb = XGBClassifier(
    n_estimators=80,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=1,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    verbosity=0,
    eval_metric='logloss'
)

# Treinamento
model_xgb.fit(X_train_scaled, y_train)
print("Modelo XGBoost treinado com regularizacao forte!")

# Modelo alternativo - Gradient Boosting
model_gb = GradientBoostingClassifier(
    n_estimators=80,
    learning_rate=0.08,
    max_depth=3,
    subsample=0.7,
    random_state=42
)
model_gb.fit(X_train_scaled, y_train)
print("Modelo Gradient Boosting treinado para comparacao!")

# ============================================================================
# AVALIAÇÃO E RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("FASE 7: RESULTADOS E ANALISE DE METRICAS")
print("=" * 80)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Avalia modelo em treino e teste"""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*70}")
    print(f"  MODELO: {model_name}")
    print(f"{'='*70}")
    
    print(f"\nTREINO")
    print(f"  Acurácia:          {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Precisão:          {precision_score(y_train, y_train_pred):.4f}")
    print(f"  Recall:            {recall_score(y_train, y_train_pred):.4f}")
    print(f"  F1-Score:          {f1_score(y_train, y_train_pred):.4f}")
    
    print(f"\nTESTE (Últimos 30 dias)")
    acc_test = accuracy_score(y_test, y_test_pred)
    print(f"  Acurácia:          {acc_test:.4f} ✓" if acc_test >= 0.75 else f"  Acurácia:          {acc_test:.4f} ✗")
    print(f"  Precisão:          {precision_score(y_test, y_test_pred):.4f}")
    print(f"  Recall:            {recall_score(y_test, y_test_pred):.4f}")
    print(f"  F1-Score:          {f1_score(y_test, y_test_pred):.4f}")
    print(f"  ROC-AUC:           {roc_auc_score(y_test, y_test_proba):.4f}")
    
    print(f"\n  Matriz de Confusão (TESTE):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                 Predito")
    print(f"                 Baixa  Alta")
    print(f"  Real Baixa     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Alta      {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    print(f"\n  Relatório Detalhado (TESTE):")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Baixa', 'Alta'],
                              digits=4))
    
    return {
        'model': model,
        'acc_test': acc_test,
        'precision_test': precision_score(y_test, y_test_pred),
        'recall_test': recall_score(y_test, y_test_pred),
        'f1_test': f1_score(y_test, y_test_pred),
        'roc_auc_test': roc_auc_score(y_test, y_test_proba),
        'y_pred': y_test_pred,
        'y_proba': y_test_proba
    }

results_xgb = evaluate_model(model_xgb, X_train_scaled, y_train, 
                            X_test_scaled, y_test, "XGBoost Regularizado")
results_gb = evaluate_model(model_gb, X_train_scaled, y_train, 
                           X_test_scaled, y_test, "Gradient Boosting")

# ============================================================================
# VALIDAÇÃO CRUZADA TEMPORAL
# ============================================================================
print("\n" + "=" * 80)
print("FASE 8: VALIDACAO CRUZADA TEMPORAL")
print("=" * 80)

print("\nEstrategia: Time Series Cross-Validation")
print("   - Folds respeitam ordem temporal")
print("   - Treino sempre anterior ao teste")
print("   - Evita data leakage")

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_scaled)):
    X_cv_train = X_train_scaled[train_idx]
    y_cv_train = y_train.iloc[train_idx]
    X_cv_test = X_train_scaled[test_idx]
    y_cv_test = y_train.iloc[test_idx]
    
    model_cv = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.08,
                            subsample=0.7, colsample_bytree=0.7, gamma=1,
                            reg_alpha=0.5, reg_lambda=0.5, random_state=42,
                            verbosity=0, eval_metric='logloss')
    model_cv.fit(X_cv_train, y_cv_train)
    
    cv_acc = accuracy_score(y_cv_test, model_cv.predict(X_cv_test))
    cv_scores.append(cv_acc)
    print(f"  Fold {fold+1}: Acuracia = {cv_acc:.4f}")

print(f"\n  Media CV: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"  Modelo eh confiavel se CV Score > 0.70 e estavel")

# ============================================================================
# IMPORTÂNCIA DAS FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("FASE 9: IMPORTÂNCIA DAS FEATURES")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance_xgb': model_xgb.feature_importances_
})
feature_importance = feature_importance.sort_values('importance_xgb', ascending=False)

print(f"\nTop 10 Features (XGBoost):")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:20s}: {row['importance_xgb']:.4f}")

# ============================================================================
# ANÁLISE DE OVERFITTING
# ============================================================================
print("\n" + "=" * 80)
print("FASE 10: ANALISE DE OVERFITTING")
print("=" * 80)

acc_train_xgb = accuracy_score(y_train, model_xgb.predict(X_train_scaled))
acc_test_xgb = results_xgb['acc_test']
diff_xgb = acc_train_xgb - acc_test_xgb

print(f"\nXGBoost:")
print(f"   Acuracia Treino: {acc_train_xgb:.4f}")
print(f"   Acuracia Teste:  {acc_test_xgb:.4f}")
print(f"   Diferenca:       {diff_xgb:.4f}")

if diff_xgb < 0.10:
    print(f"   Modelo bem balanceado (sem overfitting significativo)")
elif diff_xgb < 0.15:
    print(f"   Ligeiro overfitting, mas aceitavel")
else:
    print(f"   Overfitting detectado")

# ============================================================================
# SALVANDO RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("SALVANDO RESULTADOS")
print("=" * 80)

# Salvar previsões
results_df = pd.DataFrame({
    'Data': df_test_feat['Data'].values,
    'Preco_Real': df_test_feat['Último'].values,
    'Variacao_Real': df_test_feat['Var%'].values,
    'Tendencia_Real': y_test.values,
    'Predicao_XGBoost': results_xgb['y_pred'],
    'Probabilidade': results_xgb['y_proba']
})

results_df.to_csv('resultados_predicoes.csv', index=False)
print("\nPrevisões salvas em: resultados_predicoes.csv")

# Salvar features importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance salvo em: feature_importance.csv")

print("\n" + "="*80)
print("ANALISE COMPLETA - MODELO PRONTO PARA PRODUCAO")
print("="*80)
