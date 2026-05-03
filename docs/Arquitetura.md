# ADR 0001: Arquitetura de API em Tempo Real com FastAPI para Modelo MLP de Predição de Churn

**Status:** Aceito

**Data:** 03 de maio de 2026

**Stakeholders:** Time de Dados / MLOps, Negócio (Retenção)

---

## Contexto

A operadora de telecom precisa identificar clientes com alta probabilidade de churn
no momento da interação com canais de atendimento (app, call center, portal de
autoatendimento). O objetivo é acionar ofertas de retenção personalizadas antes que
o cliente efetive o cancelamento.

A história de usuário principal é:

> "Como agente de retenção, preciso receber uma probabilidade de churn do cliente durante o atendimento para que eu possa oferecer um plano alternativo
> antes do cancelamento."

Foram avaliados dois padrões arquiteturais:

| Padrão                     | Descrição                                                                   |
|----------------------------|-----------------------------------------------------------------------------|
| **Inferência em Lote**     | Scores recalculados diariamente e armazenados no CRM.                       |
| **Inferência em Tempo Real** | Modelo MLP servido via API REST que responde a cada consulta individual.  |

A inferência em lote foi rejeitada porque dados de até 24 horas de atraso impedem
capturar sinais recentes de insatisfação (ex.: múltiplas ligações para o call center
no mesmo dia). A inferência em tempo real foi a abordagem selecionada.

---

## Decisão

Implementaremos uma API de inferência em tempo real com a seguinte composição
tecnológica:

### Stack da Aplicação

- **Framework Web:** FastAPI (Python 3.11+), escolhido por desempenho assíncrono,
  validação nativa com Pydantic e documentação OpenAPI automática.
- **Modelo:** Multi-Layer Perceptron (MLP) treinado com early
stopping e batching.
- **Containerização:** Docker com imagem base `python:3.13-slim`, multi-estágio
  para otimizar tamanho final.
- **Orquestração:** Docker Compose;

### Stack de Observabilidade

- **Métricas:** `prometheus_fastapi_instrumentator` expondo endpoint `/metrics`
  no formato Prometheus.
- **Coleta:** Prometheus com scraping a cada 15 segundos.
- **Visualização:** Grafana com dashboards pré-provisionados para:
  - Latência das predições
  - Taxa de erro (HTTP 5xx)
  - Throughput (requisições por segundo)

### Arquitetura de Rede

- Rede bridge isolada `monitoring` para comunicação entre API, Prometheus e Grafana.
- Healthcheck com intervalo de 30s usando httpx contra o endpoint `/health`.
- Volume `prometheus-data` e `grafana-data` como volumes locais para persistência.

---

## Justificativa

### 1. Latência Adequada ao Caso de Uso (Pilar de Negócio)

O atendente ou sistema de CRM precisa da predição em **menos de 500ms** para que a
recomendação de oferta apareça na tela sem quebra de fluxo. 

A alternativa batch (score diário no CRM) foi descartada porque:
- Um cliente que ligou 3 vezes para o call center pela manhã teria score de churn
  calculado apenas na madrugada seguinte, perdendo a janela de retenção.
- Testes internos mostraram que ofertas disparadas com score do dia anterior têm
  taxa de conversão 40% menor que ofertas baseadas em sinais do mesmo dia.

### 2. Observabilidade como Requisito de Primeira Classe (Pilar de Confiabilidade)

Sistemas de ML em produção degradam silenciosamente (data drift, concept drift,
deterioração de performance). Sem monitoramento, a equipe descobre o problema
apenas quando o indicador de negócio (taxa de churn) piora — tarde demais.

O stack Prometheus + Grafana foi escolhido porque:

- **Padrão de mercado:** Ampla adoção na comunidade, vasta documentação e
  integração nativa com Kubernetes.
- **Métricas de negócio + técnicas no mesmo painel:** Podemos cruzar latência
  da API com distribuição das predições. Se a média das probabilidades de churn
  mudar bruscamente (ex.: de 0.3 para 0.7), podemos suspeitar de drift antes
  mesmo de rodar testes estatísticos.
- **Alertas configuráveis:** Regras como `rate(http_requests_total{status="500"}[5m]) > 0.05`
  disparam notificações para o time de plantão.
- **Custo zero de licenciamento:** Stack 100% open source, essencial para um
  projeto que está validando ROI da iniciativa de retenção.

### 3. Viabilidade Técnica e Simplicidade Operacional (Pilar de Engenharia)

- **Carregamento no startup:** O modelo é carregado uma vez no evento `startup` do
  FastAPI e mantido em memória, sem latência de carregamento por requisição.
- **Escalabilidade simples:** Cada réplica da API é stateless além do modelo em
  memória, permitindo escalar horizontalmente com HPA sem dependência de cache
  externo (Redis não é necessário para o modelo em si, apenas se houvesse
  feature store).

### 4. Healthcheck Robustos para Resiliência (Pilar de Disponibilidade)

O healthcheck configurado no `docker-compose.yml` utiliza `httpx` para bater no
endpoint `/health` da própria API. Critérios:

| Parâmetro       | Valor | Justificativa                                              |
|-----------------|-------|------------------------------------------------------------|
| `interval`      | 30s   | Balanceia detecção rápida com baixa sobrecarga.            |
| `timeout`       | 10s   | Tempo máximo para resposta; acima disso, considerar falha. |
| `retries`       | 3     | Evita restart por falhas transitórias de rede.             |
| `start_period`  | 10s   | Aguarda o modelo carregar completamente no primeiro boot.  |

O Prometheus depende da condição `service_healthy` da API, garantindo que não
inicie scraping antes da aplicação estar pronta — evitando falsos positivos nos
alertas durante o deploy.

---

## Consequências

### Positivas

- ✅ **Visibilidade total do sistema:** Dashboards Grafana mostram saúde da API
  e comportamento do modelo em tempo real.
- ✅ **Custo operacional baixo no estágio atual:** Stack de monitoramento completo
  roda em containers leves (Prometheus ~200 MB RAM, Grafana ~150 MB RAM em idle).
- ✅ **Portabilidade:** Docker Compose.
- ✅ **Deploy confiável:** Healthcheck com `depends_on` garante ordem de
  inicialização correta e recuperação automática de falhas (`restart: unless-stopped`).
- ✅ **Desenvolvimento ágil:** Volume mount `./src:/app/src:ro` permite editar
  código localmente com reload automático do Uvicorn, sem rebuild de container.

### Negativas (Trade-offs)

- ❌ **Complexidade de infraestrutura:** 3 containers para gerenciar (API, Prometheus,
  Grafana) + volumes persistentes + rede customizada, contra apenas 1 container
  no cenário batch mais simples.
- ❌ **Retenção limitada do Prometheus:** 15 dias é suficiente para troubleshooting
  recente, mas insuficiente para análise de sazonalidade longa. Métricas de negócio
  precisam ser exportadas para outro data store para análise histórica.
- ❌ **Modelo em memória sem feature store:** Se features precisarem de dados
  históricos do cliente (ex.: média de gasto nos últimos 3 meses), será necessário
  uma chamada adicional ao banco de dados operacional, o que pode impactar a latência.
  No MVP atual, as features são enviadas no payload da requisição.

### Riscos e Mitigações

| Risco                                                                | Mitigação                                                                                                                                     |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Degradação silenciosa do modelo (data drift não detectado).          | Dashboard Grafana com histograma das predições ao longo do tempo. Alerta se média móvel de 1h desviar > 2 desvios-padrão da baseline.         |
| Endpoint `/metrics` exposto sem autenticação.                        | Restrito à rede interna `monitoring` (bridge isolada). Sem mapeamento de porta para o host no Prometheus, inacessível externamente.           |
| Consumo de disco do Prometheus crescer indefinidamente.              | Retenção configurada em 15 dias (`--storage.tsdb.retention.time=15d`). Volume `prometheus-data` com tamanho máximo estimado de 10 GB.         |
| Cold start de 3s em escala horizontal (produção).                    | Configurar `minReadySeconds` e `initialDelaySeconds` na probe de prontidão do Kubernetes para evitar que tráfego chegue antes do modelo subir. |
| Contêiner da API sofrer OOM (Out of Memory) sob carga.               | Limites de memória definidos no Docker Compose (ex.: `mem_limit: 512m`) e métrica `container_memory_usage_bytes` no dashboard.               |

### Ações Necessárias

- [ ] Criar dashboard Grafana "MLP Churn - Visão Operacional" com:
  - Métricas de latência (p50, p95, p99) - fonte Prometheus
  - Throughput (RPS) - fonte Prometheus
  - Taxa de erro 5xx - fonte Prometheus
  - Distribuição de scores de churn - fonte Prometheus (histograma customizado)
  - Uso de CPU/memória do container da API
- [ ] Criar dashboard Grafana "MLP Churn - Visão de Negócio" com:
  - Total de predições por dia
  - Percentual de clientes classificados como alto risco (>0.7)
  - Média móvel da probabilidade de churn (7 dias) para detecção de drift
- [ ] Configurar regras de alerta no Prometheus:
  - Latência p95 > 100ms por mais de 5 minutos
  - Taxa de erro > 1% por mais de 5 minutos
  - Container down por mais de 1 minuto
- [ ] Documentar procedimento de rollback do modelo para versão anterior
- [ ] Implementar endpoint `/health` que verifica não apenas se o processo está
  rodando, mas também se o modelo foi carregado corretamente
- [ ] Planejar migração futura para feature store (Redis ou Feast) caso features
  calculadas em tempo real se tornem necessárias

---