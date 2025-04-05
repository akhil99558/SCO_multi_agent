import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DemandForecastingAgent:
    """Agent responsible for demand forecasting"""
    
    def __init__(self):
        self.model_path = 'models/demand_forecasting/random_forest_model.pkl'
        self.scaler_path = 'models/demand_forecasting/scaler.pkl'
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.model = None
            self.scaler = None
    
    def process(self, parameters):
        """Process a demand forecasting task"""
        if 'action' not in parameters:
            raise ValueError("Action parameter is required")
            
        action = parameters['action']
        
        if action == 'forecast':
            return self._generate_forecast(parameters)
        elif action == 'train':
            return self._train_model(parameters)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _generate_forecast(self, parameters):
        """Generate demand forecast for products"""
        if not self.model:
            return {
                'status': 'error',
                'message': 'Model not trained yet'
            }
        
        # Extract parameters
        product_ids = parameters.get('product_ids', [])
        store_ids = parameters.get('store_ids', [])
        forecast_horizon = parameters.get('horizon', 7)  # days
        
        # Load data
        df = pd.read_csv('data/processed/demand_features.csv')
        
        # Filter data
        if product_ids:
            df = df[df['Product ID'].isin(product_ids)]
        if store_ids:
            df = df[df['Store ID'].isin(store_ids)]
        
        # Prepare features for prediction
        features = self._prepare_forecast_features(df, forecast_horizon)
        
        # Make prediction
        X = features.drop(['Product ID', 'Store ID', 'Date'], axis=1, errors='ignore')
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Format results
        features['predicted_demand'] = predictions
        results = features[['Product ID', 'Store ID', 'Date', 'predicted_demand']]
        
        # Generate recommended actions
        actions = self._generate_actions(results)
        
        return {
            'status': 'success',
            'forecasts': results.to_dict('records'),
            'recommended_actions': actions
        }
    
    def _train_model(self, parameters):
        """Train the demand forecasting model"""
        # Load training data
        df = pd.read_csv('data/raw/demand_forecasting.csv')
        
        # Preprocess data
        df = self._preprocess_data(df)
        
        # Split features and target
        X = df.drop(['Sales Quantity', 'Product ID', 'Store ID', 'Date'], axis=1, errors='ignore')
        y = df['Sales Quantity']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=20,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Calculate training metrics
        train_score = self.model.score(X_scaled, y)
        
        return {
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': {
                'r2_score': train_score
            }
        }
    
    def _preprocess_data(self, df):
        """Preprocess raw data for model training"""
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract date features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        
        # Convert categorical features
        df = pd.get_dummies(df, columns=['Customer Segments'], drop_first=True)
        
        return df
    
    def _prepare_forecast_features(self, df, horizon):
        """Prepare features for forecasting future demand"""
        # Get the latest date in the data
        latest_date = pd.to_datetime(df['Date']).max()
        
        # Create future dates
        future_dates = pd.date_range(
            start=latest_date + pd.Timedelta(days=1),
            periods=horizon
        )
        
        # Create combinations of products, stores and dates
        product_ids = df['Product ID'].unique()
        store_ids = df['Store ID'].unique()
        
        future_df = pd.DataFrame([
            {'Product ID': p, 'Store ID': s, 'Date': d}
            for p in product_ids
            for s in store_ids
            for d in future_dates
        ])
        
        # Extract date features
        future_df['day_of_week'] = future_df['Date'].dt.dayofweek
        future_df['month'] = future_df['Date'].dt.month
        future_df['day'] = future_df['Date'].dt.day
        
        # Add other required features (e.g., price, promotions)
        # This would require more complex logic in a real system
        # For now, we'll use the most recent values for each product-store
        
        for p in product_ids:
            for s in store_ids:
                latest_record = df[(df['Product ID'] == p) & (df['Store ID'] == s)].iloc[-1]
                for col in ['Price', 'Promotions', 'Seasonality Factors']:
                    if col in df.columns:
                        future_df.loc[
                            (future_df['Product ID'] == p) & 
                            (future_df['Store ID'] == s), col
                        ] = latest_record[col]
        
        return future_df
    
    def _generate_actions(self, forecast_df):
        """Generate recommended actions based on forecasts"""
        actions = []
        
        # Load inventory data
        try:
            inventory_df = pd.read_csv('data/raw/inventory_monitoring.csv')
            
            # Merge forecast with inventory
            merged = forecast_df.merge(
                inventory_df,
                on=['Product ID', 'Store ID'],
                how='left'
            )
            
            # Identify potential stockouts
            potential_stockouts = merged[
                merged['predicted_demand'] > merged['Stock Levels']
            ]
            
            # Create reorder actions
            for _, row in potential_stockouts.iterrows():
                actions.append({
                    'action_type': 'REORDER',
                    'details': {
                        'product_id': row['Product ID'],
                        'store_id': row['Store ID'],
                        'current_stock': row['Stock Levels'],
                        'forecast_demand': row['predicted_demand'],
                        'recommended_order': max(0, row['predicted_demand'] - row['Stock Levels'])
                    },
                    'reasoning': f"Forecasted demand of {row['predicted_demand']} exceeds current stock of {row['Stock Levels']}"
                })
            
        except Exception as e:
            # If inventory data not available, don't generate actions
            pass
            
        return actions
    
    def execute_action(self, action_type, details):
        """Execute an approved action"""
        if action_type == 'REORDER':
            # In a real system, this would connect to an ordering system
            return {
                'status': 'success',
                'message': f"Order placed for {details['recommended_order']} units of product {details['product_id']} to store {details['store_id']}"
            }
        else:
            return {
                'status': 'error',
                'message': f"Action type {action_type} not supported by this agent"
            }