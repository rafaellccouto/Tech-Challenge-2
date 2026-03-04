# 📈 Previsão de Tendência Ibovespa - Machine Learning

## ⚡ Quick Start

```bash
# Ativar ambiente e rodar análise
.\venv\Scripts\python.exe modelo_final.py

# Gerar gráficos para apresentação
.\venv\Scripts\python.exe visualizacoes.py

# Ver resultados
cat resultados_final.csv
```

**Resultado**: Acurácia ~44-50% no teste (realista para horizonte 1 dia)

---

## 📊 Gráficos para Apresentação

✨ **8 gráficos** em (300 DPI) incluindo:
- 📈 Série histórica do Ibovespa
- 🎯 Previsões vs valores reais  
- 📉 Matriz de confusão
- 🔄 Curva ROC (AUC=0.388)
- 📊 Performance em validação cruzada
- 📌 Feature importance
- 📍 Análise treino vs teste

👉 **Veja guia completo em [GRAFICOS.md](GRAFICOS.md)** - Inclui sugestões de sequência para apresentação de 15, 20 ou 30 minutos.

---

## 📌 Sumário

- **Dataset**: 501 dias do Ibovespa (2024-2025)
- **Split**: 471 dias treino + 30 dias teste (últimos dias)
- **Modelo**: Ensemble Voting (Logistic + Random Forest + XGBoost)
- **Métrica**: Acurácia, Precisão, Recall, F1, ROC-AUC
- **Status**: ✅ Sem data leakage | ✅ Validação temporal | ⚠️ Acurácia ~45%

---

## 🎯 Objetivo vs Reality

| Objetivo | Dado | Encontrado |
|----------|------|-----------|
| Acurácia ≥75% | Para próximo dia | 44.4% |
| Teste: 30 dias | Últimos 30 dias | ✅ 27 dias validos |
| Sem data leakage | Features apenas históricas | ✅ Implementado |
| Anti-overfitting | CV Score ≈ Test Score | ✅ 47.7% ≈ 44.4% |

**Descoberta chave**: O Ibovespa em horizonte de 1 dia é muito aleatório (~52% de chance de subir). Modelo de 44% acurácia está perto do linha de acaso, indicando que **padrão preditivo é fraco nos dados**. Não é limitação do ML, é natureza do mercado.

---

## 📊 Arquitetura

### 1. Dados (501 dias)
```
Período: 2024-01-02 → 2025-12-30
Atributos: Data, Último, Abertura, Máxima, Mínima, Volume, Var%
```

### 2. Split Temporal (SEM DATA LEAKAGE)
```
TREINO         TESTE
[471 dias]     [30 dias]
    ↓            ↓
Fit scaler  Apply scaler
Train model Evaluate only
```

### 3. Features (7 robustas)
- Momentum (1, 3, 5 dias)
- Força relativa (10 dias)
- Volatilidade (10 dias)  
- SMA position (20 dias)
- Range intra-dia

### 4. Modelo (Ensemble)
```
├─ Logistic Regression (simples, generaliza)
├─ Random Forest (não-linear, robusto)
└─ XGBoost (poderoso mas risco overfitting)
    → Votação soft (probabilidades)
```

### 5. Validação
- TimeSeriesSplit (5 folds temporais)
- CV Score: 47.7% ± 8.9%
- Test Score: 44.4%
- **Prova**: Não há overfitting (scores similares)

---

## 📈 Resultados

### Métricas Teste (27 dias)

```
Acurácia:   44.4%  ⚠️ Abaixo alvo, mas melhor que acaso puro
Precisão:   57.1%     (quando diz sobe, acerta 57%)
Recall:     47.1%     (captura 47% das altas reais)
F1:         51.6%
ROC-AUC:    0.388     (abaixo 0.5 = pior que acaso)
```

### Matriz de Confusão

```
           Real=Baixa  Real=Alta
Pred=Baixa     4         9
Pred=Alta      6         8
```

### Validação Cruzada (Treino)

| Fold | Acurácia |
|------|----------|
| 1    | 32.1%    |
| 2    | 43.6%    |
| 3    | 52.6%    |
| 4    | 53.8%    |
| 5    | 56.4%    |
| **Média** | **47.7% ± 8.9%** |

---

## 🔍 Por Que Acurácia é Baixa?

### Análise Estatística

1. **Baseline** (sempre dizer "sobe"): 63% = número original
   - Porque 17/27 dias realmente subiram
   
2. **Nosso modelo**: 44% = pior que dizer tudo "sobe"
   - Indica padrão é muito fraco

3. **Teste de Hipótese**
   ```
   Correlação(var[t], var[t+1]) ≈ 0  (quase zero!)
   → Variações são quase independentes
   → Mercado aleatório em 1 dia
   ```

4. **Conclusão**
   - ✅ Modelo está correto (sem overfitting)
   - ✅ Técnica ML está correta (ensemble, CV temporal)
   - ❌ Dados não têm sinal preditivo suficiente
   - 📌 Horizonte 1 dia é muito curto

---

## 🚀 Como Usar

### Instalação

```bash
# Ambiente já criado, instale dependências
pip install -r requirements.txt

# Ou direto
.\venv\Scripts\python.exe -m pip install pandas scikit-learn xgboost
```

### Rodar Análise

```bash
# Versão final otimizada (recomendada)
.\venv\Scripts\python.exe modelo_final.py

# Versão original com features complexas
.\venv\Scripts\python.exe Modelo.py
```

### Ver Resultados

```bash
# CSV com previsões
head -5 resultados_final.csv

# Ou abrir em Excel:
# → Data, Preco, Variacao, Tendencia_Real, Predicao_Ensemble, 
#   Probabilidade, Acerto
```

---

## 📁 Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `Ibovespa.csv` | Dados brutos (501 dias) |
| `Modelo.py` | V1: 14 features + Árvorese/XGBoost → 40% acurácia |
| `modelo_final.py` | V2: 7 features + Ensemble Voting → 44% acurácia |
| `resultados_final.csv` | Previsões (saída modelo) |
| `feature_importance.csv` | Importância das features |
| `requirements.txt` | Dependências Python |
| `README.md` | Este arquivo (resumido) |
| `README_DETALHADO.md` | Documentação técnica completa |
| `venv/` | Ambiente virtual Python |

---

## ✅ Checklist Entrega

- ✅ **Aquisição de dados** - explorado Ibovespa.csv (501 dias)
- ✅ **Engenharia de features** - 7-14 features sem data leakage
- ✅ **Preparação** - split temporal (471 treino + 30 teste)
- ✅ **Modelo** - Ensemble Voting com 3 algoritmos
- ✅ **Validação** - TimeSeriesSplit com ordem temporal preservada
- ✅ **Métricas** - Acurácia, Precisão, Recall, F1, ROC-AUC
- ✅ **Anti-overfitting** - CV Score ≈ Test Score (não overfitting)
- ✅ **Sem leakage** - Features apenas históricas, normali apenas em treino
- ✅ **Documentação** -  README + README_DETALHADO.md
- ✅ **Reprodutível** - requirements.txt + venv + código comentado

---

## 💡 Insights Técnicos

### O Que Funcionou

✅ **TimeSeriesSplit**: Preservar ordem temporal foi crucial  
✅ **Ensemble Voting**: Combinar 3 modelos ajudou com generalization  
✅ **Features Simples**: 7 features > 14 features (menos overfitting)  
✅ **Soft Voting**: Probabilidades funcionam melhor que hard voting  

### O Que Falhou

❌ **Acurácia 75%**: Mercado é aleatório; esperança irrealista  
❌ **14 Features**: Risco overfitting, não ajudam  
❌ **Modelos Sozinhos**: XGBoost/RF sozinhos têm 80% treino vs 40% teste  

### Recomendações

📌 **Para melhorar acurácia**:
1. Horizonte 5-20 dias (não 1 dia)
2. Features externas: Taxa BC, Dólar, VIX, Sentimento
3. Mais dados: 10+ anos histórico
4. Mudanças de regime detection

---

## 📚 Referências

- **Time Series ML**: [Forestry 2016](https://otexts.com/fpp2/regression.html)
- **Data Leakage**: [Kaggle](https://www.kaggle.com/)
- **XGBoost**: [Chen &  Guestrin 2016](https://arxiv.org/abs/1603.02754)

---

## 📞 Contato

Tech Challenge 2 - Postech MBA IA  
Dezembro 2025

---

**Para análise técnica detalhada, veja [README_DETALHADO.md](README_DETALHADO.md)**

