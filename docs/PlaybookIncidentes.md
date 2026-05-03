# Playbook de Resposta a Incidentes — API de Predição de Churn (MLP)

- **Projeto:** FIAP - Predição de Churn em Telecom
- **Stack:** FastAPI + MLP (scikit-learn) + Prometheus + Grafana + Docker Compose
- **Versão do Playbook:** 1.0
- **Data:** 03 de maio de 2026
- **Times Envolvidos:** Data Science, SRE/Plataforma, Negócio (Retenção)
- **Canais de Comunicação:** Slack #alerta-churn-api, PagerDuty (plantão)

---

## Objetivo

Este playbook define os procedimentos padronizados de detecção, diagnóstico,
contenção e resolução para os alertas configurados no Prometheus. O serviço é
crítico para o fluxo de retenção de clientes — indisponibilidade superior a 5
minutos pode resultar em perda de receita por cancelamentos não interceptados.

---

## Arquitetura do Serviço


**Containers:**
- `fiap-api`: Aplicação FastAPI na porta 8000 com modelo MLP carregado em memória
- `fiap-prometheus`: Coleta métricas a cada 15s, retenção de 15 dias
- `fiap-grafana`: Dashboards operacionais e de negócio

**Rede:** Bridge isolada `monitoring` conectando os três containers.

**Healthcheck da API:**
- Comando: `python -c "import httpx; httpx.get('http://localhost:8000/health')"`
- Intervalo: 30s, Timeout: 10s, Retries: 3, Start period: 10s
- Prometheus depende de `service_healthy` da API para iniciar scraping

---

## Mapeamento de Alertas e Severidades

| Alerta | Expressão | Severidade | Gatilho | SLA Resposta |
|--------|-----------|------------|---------|-------------|
| `APIDown` | `up{job='mlp-api'} == 0` | **critical** | API fora do ar por 1m | < 5 minutos |
| `ModelNotLoaded` | `model_loaded == 0` | **high** | Modelo não carregado por 1m | < 10 minutos |
| `HighErrorRate` | `rate(5xx)/rate(total) > 5%` | **high** | > 5% erros 5xx por 2m | < 15 minutos |
| `HighLatency` | `p95 > 1s` | **medium** | Latência p95 > 1s por 5m | < 30 minutos |

**Ordem de prioridade no diagnóstico:** Sempre verifique `APIDown` primeiro, pois
pode ser a causa raiz dos demais alertas.

---

## Procedimentos por Alerta

---

### 1. APIDown (critical)

**Significado:** O endpoint `/metrics` da API não está respondendo ao scrape do
Prometheus. O container pode estar parado, em crash loop, com rede inacessível ou
com healthcheck falhando.

**Impacto no negócio:** Todas as requisições de predição estão falhando. Agentes
de retenção não recebem scores de churn. Clientes em risco de cancelamento não
são identificados. Perda financeira estimada: ofertas de retenção não disparadas.

**Diagnóstico (3 minutos):**

```bash
# 1. Verificar status de todos os containers
docker compose ps

# Saídas possíveis e interpretação:
# - fiap-api "Up" (healthy) → problema de rede entre Prometheus e API
# - fiap-api "Up" (unhealthy) → API rodando mas healthcheck falhando
# - fiap-api "Exited" → container morreu
# - fiap-api "Restarting" → crash loop
# - fiap-api não listado → serviço removido ou docker-compose.yml alterado

# 2. Verificar logs da API (últimas 50 linhas)
docker compose logs api --tail=50

# Padrões críticos nos logs:
# - "Killed" → Out of Memory (OOM kill pelo kernel)
# - "ModuleNotFoundError: No module named 'sklearn'" → dependência ausente
# - "ModuleNotFoundError: No module named 'mlp_package'" → pacote não construído
# - "Address already in use" → porta 8000 ocupada
# - "Model file not found" → arquivo .joblib ausente no container
# - "Permission denied" → problema de permissão com usuário appuser

# 3. Verificar uso de recursos do container
docker stats --no-stream fiap-api

# 4. Verificar uso de recursos do host
free -h
df -h

# 5. Testar conectividade interna pela rede monitoring
docker compose exec prometheus wget -qO- http://api:8000/health --timeout=5
echo $?

# 6. Inspecionar estado do healthcheck
docker inspect fiap-api --format='{{json .State.Health}}' | jq

# 7. Verificar logs de inicialização (erros durante bootstrap)
docker compose logs api | grep -i "error\|traceback\|failed\|cannot"

# Opção A: Reiniciar apenas o container da API (mantém rede e volumes)
docker compose restart api
echo "Aguardando start_period (10s) + healthcheck inicial..."
sleep 15
curl -s http://localhost:8000/health | jq

# Opção B: Recriar o container (se crash loop persistente)
docker compose up -d --force-recreate api
sleep 15
curl -s http://localhost:8000/health | jq

# Opção C: Rebuild da imagem (se dependências quebradas ou pacote corrompido)
docker compose build --no-cache api
docker compose up -d api
sleep 20
curl -s http://localhost:8000/health | jq

# Opção D: Se for OOM, aumentar limite de memória
# Editar docker-compose.yml e adicionar:
#   services.api.deploy:
#     resources:
#       limits:
#         memory: 512M
docker compose up -d api

# Opção E: Se porta 8000 ocupada no host
lsof -i :8000
kill -9 <PID>
docker compose restart api

# Opção F: Recriar tudo do zero (último recurso)
docker compose down -v
docker compose build --no-cache
docker compose up -d

# 1. Healthcheck da API
curl -s http://localhost:8000/health | jq

# 2. Predição de teste funcional
curl -s -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "monthly_charges": 85.5, "total_charges": 1026.0, "contract": "monthly", "payment_method": "credit_card"}' | jq

# 3. Métricas expostas
curl -s http://localhost:8000/metrics | head -20

# 4. Prometheus vendo o target como UP
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="mlp-api") | .health'

# 5. Confirmar que alerta vai cessar (forçar reload no Prometheus)
curl -X POST http://localhost:9090/-/reload
```

Se não resolver em 10 minutos:

- Notificar Product Owner de Retenção via Slack (#alerta-churn-api).
- Ativar plano de contingência: CRM exibe ofertas baseadas em regras estáticas
    (ex.: "cliente com menos de 6 meses de contrato recebe desconto de 15%").
- Escalar para Tech Lead ML e Head de Engenharia.
- Criar War Room no Slack para coordenação.

### 2. ModelNotLoaded (high)

**Significado:**  A métrica customizada model_loaded está com valor 0. A API
está rodando e respondendo, mas o modelo MLP não foi carregado na memória durante
o bootstrap. O endpoint /health deve reportar status: unhealthy com detalhe
"Model not loaded".

**Impacto no negócio:** Toda requisição ao endpoint POST /v1/predict retornará
erro 500 ou 503. Sistema efetivamente inoperante para predições, mesmo com API
respondendo a healthchecks básicos.

**Diagnóstico (3 minutos):**

```bash
# 1. Verificar resposta completa do healthcheck
curl -s http://localhost:8000/health | jq

# Respostas esperadas:
# - {"status": "healthy", "model_loaded": true, ...} → OK, falso positivo
# - {"status": "unhealthy", "detail": "Model not loaded"} → confirmado
# - {"status": "healthy", "model_loaded": false} → inconsistência na métrica

# 2. Verificar se arquivo do modelo existe e tem tamanho > 0
docker compose exec api ls -lah /app/src/models/artifacts/

# 3. Buscar logs do carregamento do modelo no startup
docker compose logs api | grep -i "model\|load\|joblib\|pth"

# 4. Verificar se o modelo existe no filesystem
docker compose exec api find /app -name "*.joblib" -o -name "*.pth" -o -name "*.sav" 2>/dev/null

# Opção A: Se arquivo existe, reiniciar API para forçar reload no startup
docker compose restart api
sleep 15
curl -s http://localhost:8000/health | jq

# Opção B: Se arquivo está faltando, verificar se foi movido/build não gerou
docker compose exec api ls -la /app/dist/
docker compose exec api ls -la /app/src/models/model_package/

# Opção C: Rebuild do pacote manualmente
docker compose exec api bash -c "cd /app/src/models/model_package && make all"
docker compose exec api pip install --force-reinstall --no-deps /app/dist/mlp_package-0.1.0-py3-none-any.whl
docker compose restart api

# Opção D: Rebuild completo da imagem (reconstroi o pacote no estágio builder)
docker compose build --no-cache api
docker compose up -d api
sleep 20

# Opção E: Verificar permissões do arquivo (deve ser legível por appuser)
docker compose exec api ls -la /app/src/models/model_package/
# Se necessário, ajustar no Dockerfile: COPY --chown=appuser:appuser ...

# 1. Confirmar métrica model_loaded == 1
curl -s http://localhost:8000/metrics | grep model_loaded

# 2. Confirmar healthcheck saudável
curl -s http://localhost:8000/health | jq

# 3. Testar predição completa
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customerID": "1234-ABCD","gender": "Female","SeniorCitizen": 1,
"Partner": "Yes","Dependents": "No","tenure": 72,"PhoneService": "Yes",
"MultipleLines": "No","InternetService": "DSL","OnlineSecurity": "Yes",
"OnlineBackup": "Yes","DeviceProtection": "Yes","TechSupport": "Yes",
"StreamingTV": "Yes","StreamingMovies": "Yes",
"Contract": "Two year","PaperlessBilling": "Yes","PaymentMethod": "Bank transfer (automatic)","MonthlyCharges": 80.85,"TotalCharges": 5821.45}' | jq

# 4. Verificar se alerta vai cessar (deve levar até 1m)
date && sleep 60 && curl -s http://localhost:8000/metrics | grep model_loaded
```

### 3. HighErrorRate (high)

**Significado:** Mais de 5% das requisições estão retornando HTTP 5xx (erro interno
do servidor) nos últimos 5 minutos, com persistência de 2 minutos. A API está
respondendo, mas uma fração significativa das chamadas está falhando.

**Impacto no negócio:** Parte dos agentes de retenção não recebe score de churn.
Dependendo do volume de chamadas, dezenas de clientes por minuto podem não ser
avaliados. Experiência do usuário inconsistente.

**Diagnóstico (5 minutos):**

```bash
# 1. Identificar quais status codes 5xx estão ocorrendo e proporção
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=sum by (status) (rate(http_requests_total{status=~"5.."}[5m]))' | jq

# Status comuns:
# - 500: Internal Server Error (exceção não tratada)
# - 502: Bad Gateway (se houver proxy reverso)
# - 503: Service Unavailable (sobrecarga ou manutenção)
# - 504: Gateway Timeout

# 2. Verificar também erros 4xx (podem indicar problema diferente)
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=sum by (status) (rate(http_requests_total{status=~"4.."}[5m]))' | jq

# 3. Buscar logs de erro recentes
docker compose logs api --tail=100 | grep -i "error\|exception\|traceback\|fail"

# 4. Testar predição manual para ver resposta completa
curl -s -w "\nHTTP_CODE: %{http_code}\nCONTENT_TYPE: %{content_type}\n" \
  -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "monthly_charges": 85.5, "total_charges": 1026.0, "contract": "monthly", "payment_method": "credit_card"}'

# 5. Testar com payload inválido para ver se causa 500 inesperado
curl -s -w "\nHTTP_CODE: %{http_code}\n" \
  -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": -1, "monthly_charges": null, "total_charges": "abc"}' | jq

# 6. Verificar volume de requisições no período do alerta
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(http_requests_total[5m])' | jq '.data.result[0].value[1]'
```

Contenção genérica (se causa não identificada rapidamente):

```bash
docker compose restart api
sleep 15
curl -s http://localhost:8000/health | jq

# Monitorar taxa de erro por 5 minutos
for i in $(seq 1 5); do
  echo "Minuto $i:"
  curl -s http://localhost:9090/api/v1/query \
    --data-urlencode 'query=sum(rate(http_requests_total{status=~"5.."}[1m])) / sum(rate(http_requests_total[1m]))' | jq
  sleep 60
done
```

### 3. HighLatency (medium)

**Significado:** O percentil 95 da latência das requisições ultrapassou 1 segundo
nos últimos 5 minutos, com persistência de 5 minutos. O SLA de 200ms está violado
para pelo menos 5% das requisições.

**Impacto no negócio:**  Agentes de retenção experimentam lentidão na tela de ofertas.
Experiência do usuário degradada mas sistema ainda funcional. Pode evoluir para
timeout e erros 504 se não tratado.

**Diagnóstico (10 minutos):**

```bash
# 1. Verificar distribuição completa de latência (p50, p95, p99)
echo "p50:"
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.50, rate(http_request_latency_seconds_bucket[5m]))' | jq '.data.result[0].value[1]'

echo "p95:"
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, rate(http_request_latency_seconds_bucket[5m]))' | jq '.data.result[0].value[1]'

echo "p99:"
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.99, rate(http_request_latency_seconds_bucket[5m]))' | jq '.data.result[0].value[1]'

# Interpretação:
# - p50 normal (< 100ms) + p95 alto → poucas requisições muito lentas (outliers)
# - p50 já alto → problema generalizado

# 2. Correlacionar com volume de requisições
echo "RPS atual:"
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(http_requests_total[5m])' | jq '.data.result[0].value[1]'

# 3. Verificar uso de CPU e memória durante o período
echo "CPU:"
docker stats --no-stream fiap-api --format "{{.CPUPerc}}"
echo "Memória:"
docker stats --no-stream fiap-api --format "{{.MemUsage}}"

# 4. Medir latência diretamente com curl
curl -w "\nTTFB: %{time_starttransfer}s\nTotal: %{time_total}s\nConnect: %{time_connect}s\n" \
  -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "monthly_charges": 85.5, "total_charges": 1026.0, "contract": "monthly", "payment_method": "credit_card"}'

# 5. Verificar se há operações bloqueantes nos logs
docker compose logs api --tail=50 --timestamps

# 6. Verificar número de workers ativos
docker compose exec api python -c "import os; print('Workers:', os.cpu_count())"
```

Contenção:

```bash
# Opção A: Aumentar workers Uvicorn (se CPU disponível)
# Editar CMD no docker-compose.yml ou Dockerfile:
# CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
docker compose restart api

# Opção B: Reiniciar container (limpa estado e memória)
docker compose restart api

# Opção C: Limitar payload máximo no FastAPI
# Adicionar no main.py:
# from fastapi import Request
# app.add_middleware(..., max_request_body_size=10_000)

# Opção D: Adicionar timeout nas chamadas externas (se houver)
# httpx.get(url, timeout=5.0)
```