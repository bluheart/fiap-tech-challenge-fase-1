# API com Rede Neural para Previsão de Churn

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.1-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-black?logo=PyTorch)](https://pytorch.org)

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Documentação](#documentação)
- [Instalação Local](#instalação-local)
- [Endpoints](#endpoints)
- [Métricas e Monitoramento](#métricas-e-monitoramento)
- [Testes](#testes)

---

## Sobre o Projeto

Esta é a **Tech Challenge** da fase 1 - produtização de modelos da Pós Tech em Engenharia de Machine Learning da FIAP.

O projeto se trata de uma API para inferência de modelos ML, com tema central: 
 Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End
- 🔥**Multi-Layer Perceptron** com PyTorch para inferência
- 📊 **Métricas Prometheus** para observabilidade
- 📝 **Logs Estruturados (JSON)** para análise
- 🔄 **Real-Time Prediction** 
- 🐳 **Stack completa** com Docker Compose (API + Prometheus + Grafana)

## Documentação
Consulte: 
- [Arquitetura](docs/Arquitetura.md)
- [ML Canvas](docs/MLCanvas.png)
- [Model Card](docs/ModelCard.md)
- [PlaybookIncidentes](docs/PlaybookIncidentes.md)


## Instalação Local

### Pré-requisitos

- Python 3.13+
- uv (como instalar [aqui](https://docs.astral.sh/uv/))
- Docker e Docker Compose
- Git

### Opção 1: Desenvolvimento Local (Python)

```bash
# Clone o repositório
git clone https://github.com/bluheart/fiap-tech-challenge-fase-1.git
cd fiap-tech-challenge-fase-1

# prepare o ambiente e dependencias
uv sync

# Ative o ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Execute a API
task test-run
```

### Opção 2: Docker Compose (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/bluheart/fiap-tech-challenge-fase-1.git
cd fiap-tech-challenge-fase-1

# Suba toda a stack
docker-compose up -d

# Acesse:
# - API:        http://localhost:8000
# - Docs:       http://localhost:8000/docs
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000 (admin/admin)
```


## Endpoints

### Públicos

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |


### Métricas

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/metrics` | Métricas Prometheus |


## Métricas e Monitoramento

### Métricas Disponíveis

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `predictions_total` | Counter | Total de predições |
| `errors_total` | Counter | Total de erros |
| `prediction_latency_seconds` | Histogram | Latência da predição |
| `model_loaded` | Gauge | Modelo está carregado |
| `current_threshold` | Gauge | Threshold da predição |

### Alertas Configurados

1. **APIDown**: API não respondendo por 1 minuto
2. **HighErrorRate**: Taxa de erro > 5% por 5 minutos
3. **HighLatency**: P95 latência > 1s por 5 minutos
4. **ModelNotLoaded**: Modelo não carregado

### Acessando Grafana

1. Acesse `http://localhost:3000`
2. Login: `admin` / `admin`
3. Adicione Data Source → Prometheus → URL: `http://prometheus:9090`
4. Importe ou crie dashboards


## Testes

### Teste de Predição

```bash

# Predição
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  {
    "customerID": "string",
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "Yes",
    "tenure": 72,
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 150,
    "TotalCharges": 0,
    "PhoneService": "Yes",
    "MultipleLines": "No phone service",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "Yes"
  }
]'
```

### Teste de Batch Prediction

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
    {
        "customerID": "1234-ABCD",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 72,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 80.85,
        "TotalCharges": 5821.45
    },
    {
        "customerID": "5678-EFGH",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.7,
        "TotalCharges": 171.4
    }
]'
```