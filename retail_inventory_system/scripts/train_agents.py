import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_demand_agent():
    """Train the demand forecasting agent"""
    print("Training Demand Forecasting Agent...")
    
    # Load data
    df = pd.read_csv('data/raw/demand_forecasting.csv')
    
    # Preprocess data
    df = preprocess_demand_data(df)
    
    # Split features and target
    # Drop unneeded columns
    X = df.drop(['Sales Quantity', 'Product ID', 'Store ID', 'Date'], axis=1, errors='ignore')

# Ensure only numeric columns remain
    X = X.select_dtypes(include=[np.number])

    y = df['Sales Quantity']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.2f}")
    
    # Save model
    model_dir = 'models/demand_forecasting'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print("  Model saved successfully!")
    
    return model, scaler

def preprocess_demand_data(df):
    """Preprocess demand forecasting data"""
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    
    # Convert categorical features
    # Identify all object or categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove 'Date' since it's already processed
    if 'Date' in categorical_cols:
        categorical_cols.remove('Date')

# Apply one-hot encoding to all categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    
    return df

def train_inventory_agent():
    """Train the inventory management agent"""
    print("Training Inventory Management Agent...")
    
    # Load data
    inventory_df = pd.read_csv('data/raw/inventory_monitoring.csv')
    demand_df = pd.read_csv('data/raw/demand_forecasting.csv')
    
    # Join data
    # Use average sales quantity per Product ID and Store ID
    sales_avg = demand_df.groupby(['Product ID', 'Store ID'])['Sales Quantity'].mean().reset_index()
    sales_avg.rename(columns={'Sales Quantity': 'Avg Sales Quantity'}, inplace=True)

# Merge average sales with inventory data
    df = pd.merge(inventory_df, sales_avg, on=['Product ID', 'Store ID'], how='left')

    
    # Feature engineering for reorder point prediction
    df = preprocess_inventory_data(df)
    
    # Split features and target
    X = df.drop(['Reorder Point', 'Product ID', 'Store ID', 'Date'], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])

    y = df['Reorder Point']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.2f}")
    
    # Save model
    model_dir = 'models/inventory_management'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print("  Model saved successfully!")
    
    return model, scaler

def preprocess_inventory_data(df):
    """Preprocess inventory data"""
    # Calculate stockout risk
    df['stockout_risk'] = df['Stockout Frequency'] * df['Avg Sales Quantity'] / df['Stock Levels']
    df['days_of_supply'] = df['Stock Levels'] / (df['Avg Sales Quantity'] / 30)

    
    # Handle missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

def train_pricing_agent():
    """Train the pricing optimization agent"""
    print("Training Pricing Optimization Agent...")
    
    # Load data
    pricing_df = pd.read_csv('data/raw/pricing_optimization.csv')
    demand_df = pd.read_csv('data/raw/demand_forecasting.csv')
    
    # Join data
    # Optional: pull only unique records for merging
    demand_cols = ['Product ID', 'Store ID', 'Promotions', 'Seasonality Factors', 'External Factors']
    demand_subset = demand_df[demand_cols].drop_duplicates(subset=['Product ID', 'Store ID'])

    df = pd.merge(pricing_df, demand_subset, on=['Product ID', 'Store ID'], how='left')

    
    # Feature engineering for elasticity prediction
    df = preprocess_pricing_data(df)
    
    # Split features and target
    X = df.drop(['Elasticity Index', 'Product ID', 'Store ID', 'Date'], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])

    y = df['Elasticity Index']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.2f}")
    
    # Save model
    model_dir = 'models/pricing_optimization'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print("  Model saved successfully!")
    
    return model, scaler

def preprocess_pricing_data(df):
    """Preprocess pricing data"""
    # Calculate price ratio
    df['price_ratio'] = df['Price'] / df['Competitor Prices']
    
    # Calculate sales efficiency
    df['sales_efficiency'] = df['Sales Volume'] / df['Storage Cost']
    
    # Calculate price-quality metric
    df['price_quality'] = df['Price'] / (df['Customer Reviews'] + 1)
    
    return df

if __name__ == "__main__":
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Train all agents
    train_demand_agent()
    train_inventory_agent()
    train_pricing_agent()
    
    print("All agents trained successfully!")