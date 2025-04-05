# 🛒 Retail Inventory Multi-Agent System: Implementation Guide

## 🧠 System Overview

This project implements a **centralized multi-agent system** for intelligent retail inventory management. The backend is powered by **Django** for API orchestration and agent coordination, while the **Streamlit** frontend provides an interactive dashboard.

The system features five specialized agents:

- 📈 **Demand Forecasting Agent**
- 📦 **Inventory Management Agent**
- 💸 **Pricing Optimization Agent**
- 🚚 **Supply Chain Coordination Agent**
- 🧍‍♂️ **Customer Behavior Agent**

---

## ⚙️ Setup Instructions

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv retail_inventory_venv
# On Windows:
retail_inventory_venv\Scripts\activate
# On macOS/Linux:
source retail_inventory_venv/bin/activate

# Install required packages
pip install django djangorestframework pandas numpy scikit-learn joblib plotly streamlit
```

### 2. Project Structure Setup

```bash
# Create necessary project directories
mkdir -p retail_inventory_system/{django_backend,streamlit_frontend,models,data/{raw,processed},scripts}

# Copy CSV files to raw data folder
cp demand_forecasting.csv inventory_monitoring.csv pricing_optimization.csv retail_inventory_system/data/raw/
```

### 3. Django Backend Setup

```bash
# Navigate to backend directory
cd retail_inventory_system/django_backend

# Create Django project
django-admin startproject retail_inventory .

# Create Django apps
python manage.py startapp agent_coordinator
python manage.py startapp api
python manage.py startapp agents

# Apply database migrations
python manage.py makemigrations
python manage.py migrate

# Run Django server
python manage.py runserver
```

### 4. Streamlit Frontend Setup

```bash
# Navigate to Streamlit directory
cd ../streamlit_frontend

# Add Streamlit UI code to app.py

# Run Streamlit dashboard
streamlit run app.py
```

---

## 🧪 Training the Agents

Train all agent models using the provided training script:

```bash
# Navigate to scripts directory
cd ../scripts

# Add training code to train_agents.py

# Execute training
python train_agents.py
```

### 🧠 Training Details

- **Models used**: RandomForestRegressor
- **Agent responsibilities**:
  - **Demand Agent**: Predicts future sales quantities
  - **Inventory Agent**: Predicts optimal reorder points
  - **Pricing Agent**: Predicts price elasticity

The script:
- Loads raw CSVs
- Applies preprocessing and feature engineering
- Trains models and evaluates them using RMSE, MAE, and R²
- Saves trained models and scalers to the `models/` directory

---

## 🔁 System Operation

### 1. Agent Coordination

The `AgentCoordinator` handles:
- Task creation and delegation
- Agent response processing
- Action recommendation and decision handling
- Execution of approved actions

### 2. User Interface (Streamlit)

The dashboard provides:
- 📊 System metrics and status
- 🔮 Demand forecasting interface
- 📦 Inventory monitoring and controls
- 💰 Price optimization recommendations
- ✅ Agent action approval panel
- 📈 Training and performance monitoring

### 3. Workflow

```text
📥 Data Collection → 📊 Forecasting → 🧠 Decision Making → ✅ Action Review → ⚙️ Execution → 🔍 Monitoring
```

- Data sourced from: `demand_forecasting.csv`, `inventory_monitoring.csv`, `pricing_optimization.csv`
- Intelligent agents suggest actions based on predictions
- Users review and approve or reject recommendations
- Approved actions are executed and tracked on the dashboard

---

## 🧩 Extending the System

To add new features or agent logic:
1. Create a new agent class in `django_backend/agents`.
2. Add new task types to the `AgentTask` model.
3. Add new action types to the `AgentAction` model.
4. Register new agents in the `AgentCoordinator`.
5. Update the UI in Streamlit to support new functionality.

---

## 📁 Directory Structure

```bash
retail_inventory_system/
├── django_backend/
│   ├── retail_inventory/         # Django project
│   ├── agent_coordinator/        # Agent coordination logic
│   ├── api/                      # REST API logic
│   └── agents/                   # Agent model classes
├── streamlit_frontend/
│   └── app.py                    # Streamlit dashboard
├── data/
│   ├── raw/                      # Raw CSV data
│   └── processed/                # Preprocessed data (if needed)
├── models/                       # Trained model and scaler files
├── scripts/
│   └── train_agents.py           # Training logic
```