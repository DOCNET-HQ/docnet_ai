# DOCNET AI Models Microservice

A FastAPI microservice for serving medical AI models with support for both JSON and image inputs, GradCAM visualizations, and AWS Lambda deployment.

## Features

- **Multi-Model Support**: Serve multiple AI models from a single service
- **GradCAM Visualization**: Automatic gradient-based class activation maps for image models
- **Flexible Input Types**: Support for both JSON and image inputs
- **AWS Lambda Ready**: Optimized for serverless deployment
- **Production-Ready**: Comprehensive error handling, logging, and monitoring
- **Framework Agnostic**: Supports TensorFlow, PyTorch, and scikit-learn models
- **RESTful API**: Clean, documented API with automatic OpenAPI/Swagger docs
- **Docker Support**: Containerized deployment option

## Project Structure

```
medical-ai-microservice/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker Compose setup
├── serverless.yml                   # AWS Lambda deployment config
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py           # Health check endpoints
│   │       └── prediction.py       # Prediction endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration settings
│   │   ├── logging_config.py       # Logging configuration
│   │   ├── exceptions.py           # Custom exceptions
│   │   └── model_registry.py       # Model loading and management
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py           # Pydantic schemas
│   └── services/
│       ├── __init__.py
│       ├── image_processor.py      # Image processing utilities
│       └── predictor.py            # Prediction service
└── models/
    ├── model_registry.json         # Model configuration file
    ├── malaria_classifier/
    │   └── model.h5
    ├── brain_tumor/
    │   └── model.h5
    └── diabetes/
        └── model.pkl
```

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd medical-ai-microservice

# Create virtual environment
python -m venv venv
source venv/bin/