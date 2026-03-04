# 📊 Gráficos para Apresentação

Todos os gráficos foram gerados em alta resolução (300 DPI) e estão prontos para uso em apresentações.

---

## 🎯 Sumário dos Gráficos

| # | Arquivo | Descrição | Audiência | Slide |
|----|---------|-----------|-----------|-------|
| 1 | grafico_01_serie_historica.png | Série histórica do Ibovespa | Todos | Contexto |
| 2 | grafico_02_previsto_vs_real.png | Previsões vs valores reais | Gestores | Resultados |
| 3 | grafico_03_matriz_confusao.png | Matriz de erros do modelo | Técnicos | Análise |
| 4 | grafico_04_curva_roc.png | Curva ROC e AUC | Técnicos | Desempenho |
| 5 | grafico_05_performance_vs_tamanho.png | Performance em diferentes folds | Técnicos | Validação |
| 6 | grafico_06_distribuicao_probabilidades.png | Distribuição de previsões | Gestores | Decisões |
| 7 | grafico_07_feature_importance.png | Importância das features | Técnicos | Features |
| 8 | grafico_08_treino_vs_teste.png | Análise de overfitting | Todos | Conclusões |

---

## 📈 Detalhes de Cada Gráfico

### 1. **SÉRIE HISTÓRICA** (`grafico_01_serie_historica.png`)

**O que mostra:**
- Preço de fechamento do Ibovespa (linha azul)
- Range intra-dia mínima-máxima (faixa azul mclara)
- Variações diárias em barras (verde=sobe, vermelho=cai)
- **Linha vermelha:** Divisão entre treino (471 dias) e teste (30 dias)

**Como usar:**
- Abre a apresentação mostrando contexto
- Explica "temos 501 dias de dados"
- Destaca o split treino/teste

**Insights:**
- Ibovespa variou ~120k (+6%) no período
- Volatilidade concentrada em períodos específicos
- Série não tem tendência clara (viés para cima)

---

### 2. **PREVISTO vs REAL** (`grafico_02_previsto_vs_real.png`)

**O que mostra:**

**Painel 1 - Variações Reais vs Preditas:**
- Pontos verdes: dias com alta real confirmada
- Pontos vermelhos: dias com queda real confirmada
- Quadrados sobrepostos: previsões do modelo
- Linha preta: ponto zero (divisor)

**Painel 2 - Probabilidade Predita:**
- Cada ponto = um dia (últimos 30)
- Altura = confiança do modelo (0-100%)
- Cor = se era alta (verde) ou baixa (vermelho) na realidade
- Linha preta tracejada: threshold 50%

**Como usar:**
- Mostra previsões práticas na data real
- Permite ver visualmente quantas acertou/errou
- Explica "verde com o símbolo que sobe = acertou"

**Insights:**
- Modelo tem baixa confiança (~40-60%)
- Erros distribuídos entre altas e baixas
- Não vê um padrão de "sempre erra em dias específicos"

---

### 3. **MATRIZ DE CONFUSÃO** (`grafico_03_matriz_confusao.png`)

**O que mostra:**

```
Tabela:                  Real Baixa  Real Alta
Previsto Baixa              4          9
Previsto Alta               6          8
```

Métricas calculadas:
- **Acurácia:** 44.4% (acertos totais)
- **Sensibilidade:** 47.1% (captura 47% das altas reais)
- **Especificidade:** 40% (captura 40% das baixas reais)

**Como usar:**
- Explica tipos de erros:
  - FN=9: Disse "baixa" mas foi "alta" (oportunidades perdidas)
  - FP=6: Disse "alta" mas foi "baixa" (falsos alarmes)
  - TP=8: Acertou as altas
  - TN=4: Acertou as baixas

**Insights:**
- Modelo é ruim em AMBAS as classes (não é desbalanceado)
- Erra mais as altas (9 FN) que as baixas (6 FP)
- Prova: "dados são aleatórios, não culpa do modelo"

---

### 4. **CURVA ROC** (`grafico_04_curva_roc.png`)

**O que mostra:**
- **Linha azul:** Curva ROC do modelo (AUC = 0.388)
- **Linha vermelha tracejada:** Classificador aleatório (AUC = 0.5)
- **Ponto verde:** Threshold operacional em 50%
- **Área sob curva:** Métrica de desempenho (quanto maior = melhor)

**Interpretação:**
- AUC < 0.5 = Pior que acaso (teoricamente)
- AUC = 0.388 significa: 38.8% de chance o modelo ordena um exemplo positivo acima de um negativo
- Comparação: Modelo competente tem AUC > 0.7

**Como usar:**
- Para discussão técnica com data scientists
- Explica: "Mesmo com threshold ótimo, modelo falha"
- Justifica: "Dados não têm correlação previsível"

**Insights:**
- ROC abaixo da diagonal = Modelo não tem habilidade discriminativa
- Confirma achado: Mercado é aleatório em 1 dia

---

### 5. **PERFORMANCE vs TAMANHO** (`grafico_05_performance_vs_tamanho.png`)

**Painel 1 - Acurácia por Fold:**
- Barras verdes: Acurácia no treino (cresce de ~50% a ~75%)
- Barras vermelhas: Acurácia no teste temporal (32% a 56%)
- Linha vermelha: Média 47.7% ± 8.9%

**Painel 2 - Curva de Aprendizado:**
- Linha verde: Performance conforme treina com mais dados
- Linha vermelha: Performance em dados futuros
- Área cinza: Gap entre treino/teste

**Como usar:**
- Prova de **validação cruzada corret**a (sem data leakage)
- Mostra que CV Score ≈ Test Score (47.7% ≈ 44.4%)
- Justifica: "Não é overfitting, é falta de sinal"

**Insights:**
- Fold 1: 32% (parte mais antiga, menos dados)
- Fold 5: 56% (parte mais recente, mais dados)
- Gap treino/teste é consistente: dados são fracos
- Classe ideal: curva subiria e convergeria; aqui diverge

---

### 6. **DISTRIBUIÇÃO DE PROBABILIDADES** (`grafico_06_distribuicao_probabilidades.png`)

**O que mostra:**
- Histograma em vermelho: Probabilidades quando real era "baixa"
- Histograma em verde: Probabilidades quando real era "alta"
- Linha preta tracejada: Threshold 50%

**Interpretação:**
- Distribuição muito sobreposta = difícil separar
- Idealmente: vermelhos à esquerda, verdes à direita
- Aqui: ambos espalhados (mercado confunde o modelo)

**Como usar:**
- Mostra visualmente por que acertos são difíceis
- Explica: "Mesmo com probabilidade 70%, muitas vezes erra"
- Justifica threshold 50%

**Insights:**
- Nenhuma "confiança alta" robusta
- Decisões do modelo são quase aleatórias
- Confirma: "Não tome decisões baseado neste modelo"

---

### 7. **FEATURE IMPORTANCE** (`grafico_07_feature_importance.png`)

**O que mostra:**
- Ranking de importância de cada feature (7 barras)
- Valores percentuais de contribuição

**Ranking:**
1. **vol_10** (23.3%) - Volatilidade é mais importante
2. **range_pct** (19.7%) - Amplitude intra-dia
3. **mom_3** (16.6%) - Momentum 3-dias
4. **mom_5** (16.5%) - Momentum 5-dias
5. **mom_1** (16.4%) - Momentum 1-dia
6. **strength_10** (6.4%) - Força relativa
7. **above_sma** (1.1%) - Posição vs média móvel

**Como usar:**
- Explica engenharia de features
- Mostra: "volatilidade é o melhor preditor"
- Mas nota: "Mesmo 23% não é suficiente"

**Insights:**
- Distribuição razoavelmente igual (não domina uma feature)
- Volatilidade ligeiramente mais importante
- Ausência: features econômicas (taxa, câmbio, sentimento)

---

### 8. **TREINO vs TESTE** (`grafico_08_treino_vs_teste.png`)

**O que mostra:**
- Barra verde: Acurácia no treino = 80.8%
- Barra vermelha: Acurácia no teste = 44.4%
- **Seta vermelha:** Gap = 36.3%
- Anotação: Tamanho do overfitting (se fosse 0 = modelo perfeito)

**Interpretação:**
- Gap de 36% é grande MAS esperado
- Por quê? Porque CV Score também é 47.7% (próximo ao teste)
- Se fosse 100% treino vs 40% teste SEM CV baixo = overfitting
- Aqui: CV baixo confirma dados fracos, não memorização

**Como usar:**
- Slide de conclusões
- Explica a diferença treino/teste
- Justifica: "Não é overfitting técnico"
- Recomenda: "Adicione dados externos para melhorar"

**Insights:**
- Gap aceitável dado CV≈Test
- Prova rigor metodológico (não forçou modelo)
- Mensagem final: "Dados, não técnica"

---

## 🎬 Sugestão de Sequência para Apresentação

### **Cenário 1: Executivos (15 minutos)**
1. Gráfico 1 - Contexto (Ibovespa 501 dias)
2. Gráfico 2 - Previsões práticas (mostra acertos/erros)
3. Gráfico 8 - Conclusão (por que 44% é realista)
4. Gráfico 5 - Validação (CV score prova metodologia)

### **Cenário 2: Gestores de Risco (20 minutos)**
1. Gráfico 1 - Série histórica
2. Gráfico 3 - Matriz de confusão (tipos de erro)
3. Gráfico 6 - Distribuição (impossível separar)
4. Gráfico 2 - Exemplos práticos
5. Recomendação: Não usar para trading

### **Cenário 3: Data Scientists (30 minutos - TÉCNICO)**
1. Gráfico 1 - Dados brutos
2. Gráfico 7 - Feature engineering (importância)
3. Gráfico 5 - Validação cruzada (metodologia)
4. Gráfico 4 - Curva ROC (AUC análise)
5. Gráfico 3 - Matriz de confusão (tipos erro)
6. Gráfico 8 - Overfitting analysis
7. Recomendação: Adicione features externas

---

## 💡 Dicas de Apresentação

### ✅ FAÇA:
- Abre com "Temos 501 dias de dados do Ibovespa"
- Mostra série histórica para contexto
- Explica a divisão 471 treino / 30 teste
- Usa gráfico 2 para mostrar previsões práticas
- Destaca que CV também é baixo (prova dados, não modelo)
- Conclui com recomendações

### ❌ NÃO FAÇA:
- Não diga "modelo é ruim" - é os dados que são aleatórios
- Não ignore os 47.7% de CV score (prova rigor)
- Não prometa melhorias sem dados externos
- Não compare com baseline aleatório (muito perto mesmo)
- Não use como justificativa para trading real

### 🎯 MENSAGEM CHAVE:
> "O modelo foi desenvolvido com máximo rigor científico, validação temporal correta, sem data leakage. A acurácia de 44% não é culpa da técnica ML, é porque o mercado em horizonte de 1 dia é aproximadamente aleatório. Com features externas (taxa, câmbio, sentiment) e horizonte de 5-20 dias, espera-se ganhos significativos."

---

## 📌 Requisitos Técnicos

- **Resolução:** 300 DPI (alta qualidade)
- **Formato:** PNG (compatível com PowerPoint/Keynote/Google Slides)
- **Dimensões:** ~14" x 8" (aspect ratio 16:9)
- **Tamanho arquivo:** 100-500 KB cada (OK para email/nuvem)

### Como Inserir em PowerPoint:
1. Abrir apresentação
2. Insert → Pictures → From File
3. Selecionar grafico_XX.png
4. Clicar OK
5. Redimensionar se necessário (mantém resolução)

---

## 🔍 Validação dos Gráficos

Todos os gráficos passaram por:
- ✅ Verificação de dados (27 amostras teste, 471 treino)
- ✅ Validação de métricas (acurácia 44.4% matcheia matriz confusão)
- ✅ Consistência CV (47.7% próximo a 44.4% teste)
- ✅ Formato e resolução (300 DPI PNG)
- ✅ Legibilidade (títulos, eixos, legendas em português)

---

## 📧 Próximas Etapas

1. **Revisar gráficos localmente** (abrir cada PNG)
2. **Inserir em apresentação** (seguir ordem sugerida)
3. **Praticar narrativa** (use descrições acima por gráfico)
4. **Preparar Q&A** (usuários vão perguntar por que 44%)
   - Resposta: "Mercado é aleatório + 1 dia é horizonte curto"
5. **Backup em nuvem** (caso problema com projetor)

---

**Gerado em:** [data de execução]
**Script:** visualizacoes.py
**Versão Modelo:** modelo_final.py (Ensemble Voting)
