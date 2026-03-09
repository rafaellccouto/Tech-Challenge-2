# Próximos Passos Checklist - Tech Challenge 2

## Status Atual ✅
- **Versão**: 2.1 (K=10 Otimizado)
- **Performance**: 81.25% acurácia ensemble, 0.80 AUC
- **Grid Search**: Concluído (K ∈ {3,5,7,10,15})
- **Documentação**: Atualizada em 9+ arquivos
- **Código**: Validado e reproduzível

---

## Fase 1: Validação (BLOQUEADOR para produção)

### ✅ Tarefa 1.1: Validação em Dados 2026
**Criticidade**: 🔴 CRÍTICA  
**Prazo**: 1 semana  
**Escopo**:
- [ ] Coletar dados Ibovespa Jan-Fev 2026
- [ ] Aplicar ensemble v2.1 (K=10)
- [ ] Calcular acurácia, AUC, métricas
- [ ] Documentar resultados em novo arquivo `VALIDACAO_2026.md`

**Success Criteria**:
- Acurácia ≥ 75% (para desbloqueiar produção)
- AUC ≥ 0.75
- Sem divergências estruturais vs Nov-Dez 2025

**Impacto**: Desbloqueia deploy para produção

**Comando para executar**:
```bash
python modelo_final.py  # Com dados 2026
```

---

## Fase 2: Otimizações (Paralelas)

### ✅ Tarefa 2.1: Otimizar Pesos do Ensemble
**Criticidade**: 🟡 ALTA  
**Prazo**: 1 semana  
**Escopo**:
- [ ] Criar `teste_pesos_óptimos.py`
- [ ] Grid search: combinações de [w_lr, w_rf, w_xgb, w_knn]
- [ ] Testar com TimeSeriesSplit (mesmo protocolo)
- [ ] Comparar vs pesos atuais [1, 1.2, 1.5, 0.8]

**Success Criteria**:
- Encontrar combinação com acurácia > 81.25%
- Documentar em `ATUALIZACAO_PESOS.md`

**Possível Ganho**: +1-2% acurácia  
**Impacto**: Refinamento fino do ensemble

**Arquivo chave**: `modelo_final.py` linha ~290
```python
voting_clf = VotingClassifier(
    estimators=[...],
    weights=[1, 1.2, 1.5, 0.8]  # ← OTIMIZAR AQUI
)
```

### ✅ Tarefa 2.2: Feature Selection (Redimensionalidade)
**Criticidade**: 🟡 ALTA  
**Prazo**: 2 semanas  
**Escopo**:
- [ ] Analisar importância de features (RF/XGB)
- [ ] Selecionar top 5-7 features
- [ ] Recriar modelos com features reduzidas
- [ ] Comparar performance vs 11 features atuais

**Success Criteria**:
- KNN AUC sobe de 0.75 → 0.80+
- Ensemble mantém ≥ 81.25% acurácia
- Tempo treino reduzido

**Possível Ganho**: +1-2% em KNN, eliminação curse of dimensionality  
**Impacto**: Melhoria KNN, dados mais interpretáveis

**Candidates para remover**:
- Features colineares (MM5 ≈ MM10?)
- Features com baixa importância

---

## Fase 3: Expansão (Macro)

### ✅ Tarefa 3.1: Adicionar Features Exógenas
**Criticidade**: 🟡 ALTA  
**Prazo**: 3 semanas  
**Escopo**:
- [ ] Coletar dados históricos:
  - USD/BRL (taxa cambial)
  - Selic rate (taxa de juros)
  - VIX (volatilidade global)
- [ ] Sincronizar datas com Ibovespa
- [ ] Adicionar como features ao pipeline
- [ ] Testar ensemble com dados macro

**Success Criteria**:
- CV score sobe de ~50% → 55%+
- Séries autocorrelação melhora
- Abordagem mais robusta

**Por quê**: Séries temporal fraca (ρ≈-0.05) sugere falta de drivers externos  
**Impacto**: Fundamental para robustez em produção

**Dados necessários**:
```python
features_new = {
    'USDBRL': [...],    # Cambial
    'SELIC': [...],     # Taxa juros
    'VIX': [...]        # Volatilidade
}
```

---

## Fase 4: Exploração (Futura)

### ✅ Tarefa 4.1: Estender Horizonte de Previsão
**Criticidade**: 🟢 BAIXA  
**Prazo**: 4+ semanas  
**Escopo**:
- [ ] Testar 5-day trend prediction
- [ ] Testar 20-day trend prediction
- [ ] Comparar vs atual 1-day
- [ ] Analisar trade-off sinal/ruído

**Success Criteria**:
- Horizonte 5-day: acurácia ≥ 75%
- Horizonte 20-day: acurácia ≥ 70%
- Sinais mais claros (menor noise)

**Ganho esperado**: Sinais mais fortes, menos whiplash  
**Impacto**: Melhoria operacional em produção

---

## Checklist de Verificação

### Antes de Deploy em Produção
- [ ] Validação 2026 completada (Tarefa 1.1)
- [ ] Acurácia 2026 ≥ 75%
- [ ] Código `modelo_final.py` testado end-to-end
- [ ] Documentação `VALIDACAO_2026.md` criada
- [ ] Todos os arquivos `.py` e `.csv` salvos
- [ ] Gráficos 2026 gerados (PNG 300 DPI)

### Depois de Deploy (Monitoramento)
- [ ] Executar Tarefa 2.1 (pesos) — +1-2% possível
- [ ] Executar Tarefa 2.2 (features) — melhoria KNN
- [ ] Executar Tarefa 3.1 (macro) — estabilidade
- [ ] Executar Tarefa 4.1 (horizonte) — exploração

---

## Histórico de Decisões

| Data | Decisão | Racional | Impacto |
|------|---------|----------|---------|
| 2026-03-04 | Escolher KNN K=5 | Erro inicial | -36% AUC |
| 2026-03-09 | Grid search K ∈ {3..15} | Otimização | K=10 melhor |
| 2026-03-09 | Adotar K=10 | AUC 0.75 vs 0.55 | +12.5% ensemble |
| 2026-03-09 | Deploy v2.1 K=10 | Meta atingida | 81.25% < 75%? |
| 2026-06-xx | Validar 2026 | Verificar estabilidade | **PRÓXIMO** |

---

## Recursos e Arquivos-Chave

### Código
- `modelo_final.py` — Pipeline v2.1 (K=10)
- `teste_knn_k_otimo.py` — Grid search script
- `teste_pesos_óptimos.py` — **A CRIAR**
- `visualizacoes.py` — Gerador de gráficos

### Documentação
- `SUMMARY.md` — Storytelling principal (ATUALIZADO)
- `RESUMO_EXECUTIVO.md` — Executivo (ATUALIZADO)
- `ATUALIZACAO_K10.md` — K=10 optimization summary
- `VALIDACAO_2026.md` — **A CRIAR**
- `ATUALIZACAO_PESOS.md` — **A CRIAR**

### Dados
- `Ibovespa.csv` — Input Nov-Dez 2025
- `resultados_knn_k_otimo.csv` — Grid search results
- Dados 2026 — **A COLETAR S/N**

---

## Contatos e Escalação

| Tarefa | Responsável | Prioridade | Contato |
|--------|------------|-----------|---------|
| Validação 2026 | Data Engineer | 🔴 BLOQUEADOR | - |
| Pesos ensemble | Data Scientist | 🟡 ALTA | - |
| Features exógenas | Data Engineer | 🟡 ALTA | - |
| Deploy produção | DevOps | 🟡 ALTA | - |

---

**Última Atualização**: 2026-03-09 (K=10 grid search concluído)  
**Próxima Checkpoint**: 2026-06-04 (Meta validation 2026)  
**Status**: ✅ PRONTO PARA FASE 1 (Validação 2026)
