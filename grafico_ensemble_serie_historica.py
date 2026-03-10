"""
Gráfico Detalhado: Série Histórica do Ensemble com Dados Reais
Mostra a performance do ensemble em comparação com o movimento real do Ibovespa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Cores customizadas
CORES = {
    'real': '#1f77b4',        # Azul
    'ensemble': '#d62728',    # Vermelho
    'sobe': '#2ca02c',        # Verde
    'desce': '#ff7f0e',       # Laranja
    'acerto': '#2ca02c',      # Verde
    'erro': '#d62728'         # Vermelho
}

print("=" * 80)
print("SÉRIE HISTÓRICA DO ENSEMBLE COM DADOS REAIS")
print("=" * 80)

# ============================================================
# 1. CARREGAR E PROCESSAR DADOS
# ============================================================

print("\n[1/5] Carregando dados...")

df = pd.read_csv('Ibovespa.csv')
print(f"✓ {len(df)} dias carregados")

# Renomear colunas
df = df.rename(columns={
    'Último': 'Ultimo',
    'Máxima': 'Maxima',
    'Mínima': 'Minima',
    'Vol.': 'Vol',
    'Var%': 'VarPerc'
})

# Converter data
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')

# Processar números (locale português: pontos como separador de milhares, vírgulas como decimais)
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

# Criar target
df['Tendencia'] = (df['Ultimo'].shift(-1) > df['Ultimo']).astype(int)

# Remover última linha (sem target) e NaNs
df = df[:-1]
df = df.dropna()
print(f"✓ {len(df)} registros válidos após processamento")

# Criar target
df['Tendencia'] = (df['Ultimo'].shift(-1) > df['Ultimo']).astype(int)
df = df[:-1]  # Remover último (sem target)

# Features
X = df[['Abertura', 'Maxima', 'Minima', 'Vol', 'VarPerc']].values

# Split temporal (últimos 44 dias para teste)
split_point = len(df) - 44
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = df['Tendencia'].values[:split_point], df['Tendencia'].values[split_point:]
dates_test = df['Data'].values[split_point:]

print(f"✓ Split temporal: {len(X_train)} treino, {len(X_test)} teste")

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 2. TREINAR ENSEMBLE
# ============================================================

print("\n[2/5] Treinando ensemble...")

lr = LogisticRegression(random_state=42, max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
knn = KNeighborsClassifier(n_neighbors=10)

voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_model), ('knn', knn)],
    voting='soft',
    weights=[1.0, 1.2, 1.5, 0.8]
)

voting_clf.fit(X_train_scaled, y_train)
print("✓ Ensemble treinado com sucesso")

# Previsões
y_pred_ensemble = voting_clf.predict(X_test_scaled)
y_proba_ensemble = voting_clf.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# 3. CRIAR FIGURA COM MÚLTIPLOS SUBPLOTS
# ============================================================

print("\n[3/5] Gerando gráficos...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

# ============================================================
# PLOT 1: SÉRIE HISTÓRICA COMPLETA (Real vs Ensemble)
# ============================================================

ax1 = fig.add_subplot(gs[0, :])

dias = np.arange(len(y_test))
width = 0.35

# Preparar dados para barras
bars_real = []
bars_pred = []
colors_real = []
colors_pred = []

for i in range(len(y_test)):
    if y_test[i] == 1:
        bars_real.append(1)
        colors_real.append(CORES['sobe'])
    else:
        bars_real.append(0)
        colors_real.append(CORES['desce'])
    
    if y_pred_ensemble[i] == 1:
        bars_pred.append(1)
        colors_pred.append(CORES['sobe'])
    else:
        bars_pred.append(0)
        colors_pred.append(CORES['desce'])

x_pos = np.arange(len(dias))
ax1.bar(x_pos - width/2, bars_real, width, label='Real', alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.bar(x_pos + width/2, bars_pred, width, label='Ensemble (Previsto)', alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Dia de Teste', fontweight='bold', fontsize=12)
ax1.set_ylabel('Movimento', fontweight='bold', fontsize=12)
ax1.set_title('Série Histórica Completa: Real vs Ensemble (44 dias de teste)', fontweight='bold', fontsize=14)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['DESCE', 'SOBE'])
ax1.set_xticks(x_pos[::5])
ax1.set_xticklabels([f'Dia {i+1}' for i in x_pos[::5]])
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Adicionar taxa de acerto
acertos = (y_pred_ensemble == y_test).sum()
total = len(y_test)
accuracy = acertos / total
ax1.text(0.02, 0.95, f'Taxa de Acerto: {accuracy:.1%} ({acertos}/{total} dias)',
        transform=ax1.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================
# PLOT 2: ZOOM - ÚLTIMOS 20 DIAS
# ============================================================

ax2 = fig.add_subplot(gs[1, 0])

zoom_range = min(20, len(y_test))
zoom_start = len(y_test) - zoom_range
dias_zoom = np.arange(zoom_range)

bars_real_zoom = [y_test[zoom_start + i] for i in range(zoom_range)]
bars_pred_zoom = [y_pred_ensemble[zoom_start + i] for i in range(zoom_range)]

x_zoom = np.arange(zoom_range)
ax2.bar(x_zoom - width/2, bars_real_zoom, width, label='Real', alpha=0.8, color=CORES['real'], edgecolor='black', linewidth=0.5)
ax2.bar(x_zoom + width/2, bars_pred_zoom, width, label='Ensemble', alpha=0.8, color=CORES['ensemble'], edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Dia de Teste', fontweight='bold', fontsize=11)
ax2.set_ylabel('Movimento', fontweight='bold', fontsize=11)
ax2.set_title(f'Zoom: Últimos {zoom_range} Dias', fontweight='bold', fontsize=12)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['DESCE', 'SOBE'])
ax2.set_xticks(x_zoom)
ax2.set_xticklabels([f'{i+1}' for i in x_zoom], fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# ============================================================
# PLOT 3: CONFIANÇA DO ENSEMBLE (Probabilidades)
# ============================================================

ax3 = fig.add_subplot(gs[1, 1])

ax3.fill_between(dias, 0.5, y_proba_ensemble, 
                where=(y_proba_ensemble >= 0.5), alpha=0.3, color=CORES['sobe'], label='Confiante em SOBE')
ax3.fill_between(dias, 0.5, y_proba_ensemble, 
                where=(y_proba_ensemble < 0.5), alpha=0.3, color=CORES['desce'], label='Confiante em DESCE')
ax3.plot(dias, y_proba_ensemble, 'o-', color=CORES['ensemble'], linewidth=2, markersize=5, label='Probabilidade Ensemble')
ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (50%)')

# Colorir fundo pela realidade
for i in range(len(y_test)):
    if y_test[i] == 1:
        ax3.axvspan(i-0.5, i+0.5, alpha=0.05, color=CORES['sobe'])
    else:
        ax3.axvspan(i-0.5, i+0.5, alpha=0.05, color=CORES['desce'])

ax3.set_xlabel('Dia de Teste', fontweight='bold', fontsize=11)
ax3.set_ylabel('Probabilidade', fontweight='bold', fontsize=11)
ax3.set_title('Confiança do Ensemble (Probabilidade de SOBE)', fontweight='bold', fontsize=12)
ax3.set_ylim([0, 1])
ax3.set_xticks(dias[::5])
ax3.set_xticklabels([f'Dia {i+1}' for i in dias[::5]], fontsize=9)
ax3.legend(fontsize=9, loc='best')
ax3.grid(alpha=0.3)

# ============================================================
# PLOT 4: ACERTOS vs ERROS (Stacked View)
# ============================================================

ax4 = fig.add_subplot(gs[2, 0])

acertos = (y_pred_ensemble == y_test).astype(int)
colors = [CORES['acerto'] if x == 1 else CORES['erro'] for x in acertos]

bars = ax4.bar(dias, acertos, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax4.set_xlabel('Dia de Teste', fontweight='bold', fontsize=11)
ax4.set_ylabel('Acertou?', fontweight='bold', fontsize=11)
ax4.set_title('Acertos (Verde) vs Erros (Vermelho)', fontweight='bold', fontsize=12)
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['Erro', 'Acerto'])
ax4.set_xticks(dias[::5])
ax4.set_xticklabels([f'Dia {i+1}' for i in dias[::5]], fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# Adicionar estatísticas
erros = (acertos == 0).sum()
ax4.text(0.98, 0.95, f'Erros: {erros} dias\nAcertos: {acertos.sum()} dias',
        transform=ax4.transAxes, fontsize=11, fontweight='bold',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================
# PLOT 5: COMPARAÇÃO REAL vs PREVISTO (Com Datas)
# ============================================================

ax5 = fig.add_subplot(gs[2, 1])

# Usando datas como x
dates_str = [pd.Timestamp(d).strftime('%d/%m') for d in dates_test]
x_dates = np.arange(len(dates_str))

ax5.plot(x_dates, y_test, 'o-', label='Real', color=CORES['real'], linewidth=2.5, markersize=7, alpha=0.9)
ax5.plot(x_dates, y_pred_ensemble, 's--', label='Ensemble', color=CORES['ensemble'], linewidth=2, markersize=6, alpha=0.8)

ax5.set_xlabel('Data', fontweight='bold', fontsize=11)
ax5.set_ylabel('Movimento (0=DESCE, 1=SOBE)', fontweight='bold', fontsize=11)
ax5.set_title('Série Histórica com Datas Reais', fontweight='bold', fontsize=12)
ax5.set_yticks([0, 1])
ax5.set_yticklabels(['DESCE', 'SOBE'])
ax5.set_xticks(x_dates[::5])
ax5.set_xticklabels([dates_str[i] for i in x_dates[::5]], fontsize=9, rotation=45)
ax5.legend(fontsize=11, loc='best')
ax5.grid(alpha=0.3)

# Adicionar linha conexão aos erros
for i in range(len(y_test)):
    if acertos[i] == 0:
        ax5.scatter(i, y_test[i], s=200, marker='x', color='red', linewidth=3, zorder=5)

plt.suptitle('ENSEMBLE - SÉRIE HISTÓRICA COM DADOS REAIS\nAnálise Detalhada de 44 Dias de Teste (Nov-Dez 2025)',
            fontsize=16, fontweight='bold', y=0.995)

# Salvar
plt.savefig('ensemble_serie_historica_detalhada.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: ensemble_serie_historica_detalhada.png")
plt.close()

# ============================================================
# 4. CRIAR GRÁFICO ADICIONAL: RETORNO SIMULADO
# ============================================================

print("[4/5] Gerando gráfico de retorno simulado...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Extrair preços reais
precos_reais = df['Ultimo'].values[split_point:split_point+len(y_test)]

# PLOT 1: SÉRIE DE PREÇOS COM DECISÕES DO ENSEMBLE
ax = axes[0]

ax.plot(dias, precos_reais, 'o-', color='black', linewidth=2.5, markersize=6, label='Preço Real Ibovespa', alpha=0.8)

# Colorir fundo por previsão correta/incorreta
for i in range(len(y_test)):
    if acertos[i] == 1:
        ax.axvspan(i-0.5, i+0.5, alpha=0.1, color=CORES['acerto'])
    else:
        ax.axvspan(i-0.5, i+0.5, alpha=0.15, color=CORES['erro'])

# Adicionar marcadores de decisão
for i in range(len(y_test)):
    if y_pred_ensemble[i] == 1:
        ax.scatter(i, precos_reais[i], marker='^', s=150, color=CORES['sobe'], edgecolor='black', linewidth=1, zorder=5, alpha=0.7)
    else:
        ax.scatter(i, precos_reais[i], marker='v', s=150, color=CORES['desce'], edgecolor='black', linewidth=1, zorder=5, alpha=0.7)

ax.set_xlabel('Dia de Teste', fontweight='bold', fontsize=12)
ax.set_ylabel('Preço Ibovespa', fontweight='bold', fontsize=12)
ax.set_title('Série de Preços com Decisões do Ensemble\n▲ = Prevê SOBE | ▼ = Prevê DESCE | Verde = Acerto | Vermelho = Erro',
            fontweight='bold', fontsize=13)
ax.set_xticks(dias[::5])
ax.set_xticklabels([f'Dia {i+1}' for i in dias[::5]], fontsize=10)
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)

# PLOT 2: RETORNO ACUMULADO SIMULADO
ax = axes[1]

# Simular retorno: +1% se acerta SOBE, -1% se acerta DESCE, -0.5% se erra
retorno_diario = []
for i in range(len(y_test)):
    if acertos[i] == 1:  # Acertou
        if y_test[i] == 1:  # Era SOBE
            retorno_diario.append(0.01)  # Ganho 1%
        else:  # Era DESCE
            retorno_diario.append(0.01)  # Ganho 1% (vendido curto)
    else:  # Errou
        retorno_diario.append(-0.005)  # Perda 0.5%

retorno_acumulado = np.cumprod(1 + np.array(retorno_diario)) - 1

ax.plot(dias, retorno_acumulado, 'o-', color=CORES['ensemble'], linewidth=3, markersize=7, label='Retorno Acumulado (Ensemble)', alpha=0.9)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Preencher positivo/negativo
ax.fill_between(dias, 0, retorno_acumulado, where=(retorno_acumulado >= 0), alpha=0.3, color=CORES['sobe'], label='Ganho')
ax.fill_between(dias, 0, retorno_acumulado, where=(retorno_acumulado < 0), alpha=0.3, color=CORES['desce'], label='Perda')

ax.set_xlabel('Dia de Teste', fontweight='bold', fontsize=12)
ax.set_ylabel('Retorno Acumulado (%)', fontweight='bold', fontsize=12)
ax.set_title('Retorno Simulado Seguindo Previsões do Ensemble\n(+1% por acerto, -0.5% por erro)',
            fontweight='bold', fontsize=13)
ax.set_xticks(dias[::5])
ax.set_xticklabels([f'Dia {i+1}' for i in dias[::5]], fontsize=10)
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)

# Adicionar métrica final
retorno_final = retorno_acumulado[-1] * 100
ax.text(0.98, 0.95 if retorno_final > 0 else 0.05, f'Retorno Final: {retorno_final:.2f}%',
       transform=ax.transAxes, fontsize=12, fontweight='bold',
       verticalalignment='top' if retorno_final > 0 else 'bottom',
       horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('ensemble_serie_historica_precos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: ensemble_serie_historica_precos.png")
plt.close()

# ============================================================
# 5. RESUMO FINAL
# ============================================================

print("\n[5/5] Gerando resumo...")
print("\n" + "=" * 80)
print("RESUMO - SÉRIE HISTÓRICA DO ENSEMBLE")
print("=" * 80)

print(f"\n✓ Acurácia Ensemble:          {accuracy:.1%}")
print(f"✓ Dias Corretos:             {acertos.sum()} / {total}")
print(f"✓ Dias Incorretos:           {total - acertos.sum()}")
print(f"✓ Probabilidade Média SOBE:  {y_proba_ensemble.mean():.1%}")
print(f"✓ Variância Probabilidade:   {np.std(y_proba_ensemble):.3f}")
print(f"✓ Retorno Simulado Final:    {retorno_final:.2f}%")

print(f"\n✓ Data Início Teste:         {pd.Timestamp(dates_test[0]).strftime('%d/%m/%Y')}")
print(f"✓ Data Fim Teste:            {pd.Timestamp(dates_test[-1]).strftime('%d/%m/%Y')}")
print(f"✓ Preço Inicial Ibovespa:    {precos_reais[0]:,.2f}")
print(f"✓ Preço Final Ibovespa:      {precos_reais[-1]:,.2f}")
print(f"✓ Variação Preço Real:       {(precos_reais[-1] / precos_reais[0] - 1) * 100:.2f}%")

print("\n" + "=" * 80)
print("✅ GRÁFICOS GERADOS COM SUCESSO!")
print("=" * 80)
print("\nArquivos criados:")
print("  1. ensemble_serie_historica_detalhada.png  (6 subplots analíticos)")
print("  2. ensemble_serie_historica_precos.png     (preços + retorno simulado)")
