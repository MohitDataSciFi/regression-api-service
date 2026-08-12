# regression-api-service

Production-grade FastAPI service for training and serving regression models with validation, logging, and tests.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Overview

A production-ready REST API service built with FastAPI that provides end-to-end regression model management. The service handles model training, serialization, and prediction serving with comprehensive validation, structured logging, and automated testing. Designed for data science teams requiring a robust, scalable solution for deploying regression models in production environments.

## Tech Stack

- **FastAPI** — Modern, high-performance web framework for building APIs
- **Pydantic** — Data validation and settings management using Python type annotations
- **scikit-learn** — Machine learning library for regression model implementation
- **statsmodels** — Statistical modeling and hypothesis testing
- **joblib** — Efficient model serialization and deserialization
- **pytest** — Testing framework for unit and integration tests
- **uvicorn** — ASGI server for production deployment
- **numpy** — Numerical computing and array operations

## Multi-Phase Roadmap

### Phase 1: Core Model Training and Serialization
Implement a regression model manager that trains an OLS model using scikit-learn and statsmodels, computes diagnostics (R², coefficients, p-values), and serializes the model with joblib. Expose a training endpoint that accepts CSV data via multipart upload, validates with Pydantic, and returns training metrics. Set up structured logging and basic error handling.

### Phase 2: Prediction API, Testing, and Deployment Readiness
Add a prediction endpoint that loads the serialized model and returns predictions with confidence intervals. Implement comprehensive input validation using Pydantic models, add unit tests with pytest covering training and prediction flows, and include Dockerfile and uvicorn configuration for production deployment. Ensure graceful handling of missing model and invalid inputs with appropriate HTTP status codes.

## Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/regression-api-service.git
cd regression-api-service

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Service

```bash
# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive API documentation can be accessed at `http://localhost:8000/docs`.

### Testing

```bash
# Run the test suite
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the existing style conventions and includes appropriate test coverage.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.