import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# API configuration
API_BASE_URL = "http://localhost:8000/api"

def main():
    st.set_page_config(
        page_title="Retail Inventory Management",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Retail Inventory Multi-Agent System")
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Forecasting", "Inventory", "Pricing", "Actions", "Agent Training"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Forecasting":
        show_forecasting()
    elif page == "Inventory":
        show_inventory()
    elif page == "Pricing":
        show_pricing()
    elif page == "Actions":
        show_actions()
    elif page == "Agent Training":
        show_agent_training()

def show_dashboard():
    """Display the main dashboard"""
    st.header("System Overview")
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Products", "1,245", "+12")
    
    with col2:
        st.metric("Pending Actions", "24", "+5")
    
    with col3:
        st.metric("Forecast Accuracy", "92.4%", "+1.2%")
    
    with col4:
        st.metric("Stockout Rate", "2.1%", "-0.5%")
    
    # Get recent tasks
    tasks = get_recent_tasks()
    
    # Display recent tasks
    st.subheader("Recent Agent Tasks")
    if tasks:
        task_df = pd.DataFrame(tasks)
        st.dataframe(task_df[['id', 'task_type', 'status', 'created_at']])
    else:
        st.info("No recent tasks found")
    
    # Display stock alerts
    st.subheader("Stock Alerts")
    show_stock_alerts()
    
    # Agent activity chart
    st.subheader("Agent Activity")
    show_agent_activity()

def show_forecasting():
    """Display demand forecasting interface"""
    st.header("Demand Forecasting")
    
    # Input form
    with st.form("forecast_form"):
        st.subheader("Generate New Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            products = st.multiselect(
                "Select Products",
                options=get_product_list(),
                default=get_product_list()[:5]
            )
        
        with col2:
            stores = st.multiselect(
                "Select Stores",
                options=get_store_list(),
                default=get_store_list()[:3]
            )
        
        horizon = st.slider("Forecast Horizon (Days)", 1, 30, 7)
        
        submitted = st.form_submit_button("Generate Forecast")
        
        if submitted:
            with st.spinner("Generating forecast..."):
                # Create forecast task
                task = create_forecast_task(products, stores, horizon)
                
                if task and task.get('status') != 'FAILED':
                    st.success(f"Forecast task created successfully! Task ID: {task['id']}")
                else:
                    st.error("Failed to create forecast task")
    
    # Display recent forecasts
    st.subheader("Recent Forecasts")
    show_recent_forecasts()

def show_inventory():
    """Display inventory management interface"""
    st.header("Inventory Management")
    
    tab1, tab2 = st.tabs(["Current Inventory", "Reorder Suggestions"])
    
    with tab1:
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            store_filter = st.selectbox(
                "Filter by Store",
                options=["All Stores"] + get_store_list()
            )
        
        with col2:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All", "Low Stock", "Optimal", "Overstocked"]
            )
        
        # Display inventory table
        inventory_data = get_inventory_data(store_filter, status_filter)
        
        if inventory_data.empty:
            st.info("No inventory data found matching the filters")
        else:
            st.dataframe(inventory_data)
    
    with tab2:
        # Display reorder suggestions
        reorder_data = get_reorder_suggestions()
        
        if reorder_data.empty:
            st.info("No reorder suggestions available")
        else:
            st.dataframe(reorder_data)
            
            # Batch approve
            if st.button("Approve All Suggestions"):
                with st.spinner("Approving suggestions..."):
                    # Simulate API call
                    st.success("All suggestions approved!")

def show_pricing():
    """Display pricing optimization interface"""
    st.header("Pricing Optimization")
    
    # Display price optimization suggestions
    pricing_data = get_pricing_suggestions()
    
    if not pricing_data.empty:
        st.subheader("Suggested Price Changes")
        
        # Display price change table
        st.dataframe(pricing_data)
        
        # Price elasticity visualization
        st.subheader("Price Elasticity Analysis")
        fig = create_elasticity_chart(pricing_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No pricing suggestions available")
    
    # Form for custom price analysis
    with st.expander("Custom Price Analysis"):
        with st.form("price_analysis_form"):
            st.subheader("Analyze Price Changes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                product = st.selectbox(
                    "Select Product",
                    options=get_product_list()
                )
            
            with col2:
                price_change = st.slider(
                    "Price Change (%)",
                    min_value=-20,
                    max_value=20,
                    value=0,
                    step=1
                )
            
            submitted = st.form_submit_button("Analyze Impact")
            
            if submitted:
                with st.spinner("Analyzing price impact..."):
                    # Simulate analysis
                    st.success("Analysis complete!")
                    
                    # Show results
                    display_price_impact_analysis(product, price_change)

def show_actions():
    """Display agent recommendations and actions"""
    st.header("Agent Recommendations")
    
    # Get pending actions
    actions = get_pending_actions()
    
    if actions:
        for action in actions:
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"{action['action_type']} Action")
                    st.write(f"**Reasoning:** {action['reasoning']}")
                    
                    # Display details
                    st.json(action['details'])
                
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("Approve", key=f"approve_{action['id']}"):
                        with st.spinner("Approving action..."):
                            # Simulate API call
                            approve_action(action['id'])
                            st.success("Action approved!")
                            st.rerun()
                    
                    if st.button("Reject", key=f"reject_{action['id']}"):
                        reason = st.text_input(
                            "Rejection reason",
                            key=f"reason_{action['id']}"
                        )
                        if st.button("Confirm Rejection", key=f"confirm_{action['id']}"):
                            with st.spinner("Rejecting action..."):
                                # Simulate API call
                                reject_action(action['id'], reason)
                                st.success("Action rejected!")
                                st.rerun()
                
                st.divider()
    else:
        st.info("No pending actions found")
    
    # Show action history
    st.subheader("Action History")
    show_action_history()

def show_agent_training():
    """Display agent training interface"""
    st.header("Agent Training")
    
    tab1, tab2 = st.tabs(["Train Models", "Performance Metrics"])
    
    with tab1:
        # Agent selection
        agent_type = st.selectbox(
            "Select Agent to Train",
            options=[
                "Demand Forecasting Agent",
                "Inventory Management Agent",
                "Pricing Optimization Agent",
                "Supply Chain Agent",
                "Customer Behavior Agent"
            ]
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            training_data = st.file_uploader(
                "Upload Training Data (Optional)",
                type=["csv"]
            )
            
            use_default = st.checkbox(
                "Use Default Dataset",
                value=True
            )
        
        with col2:
            # Model parameters
            if agent_type == "Demand Forecasting Agent":
                model_type = st.selectbox(
                    "Model Type",
                    options=["Random Forest", "LSTM", "Prophet"]
                )
                
                if model_type == "Random Forest":
                    n_estimators = st.slider(
                        "Number of Estimators",
                        min_value=10,
                        max_value=500,
                        value=100,
                        step=10
                    )
        
        # Submit training job
        if st.button("Start Training"):
            with st.spinner("Training agent..."):
                # Simulate API call
                st.success(f"{agent_type} trained successfully!")
                
                # Show metrics after training
                show_training_metrics(agent_type)
    
    with tab2:
        # Display agent performance metrics
        show_agent_performance_metrics()

# Helper functions

def get_recent_tasks():
    """Get recent tasks from API"""
    # In a real implementation, this would make an API call
    # For demo purposes, return dummy data
    return [
        {
            "id": 1,
            "task_type": "FORECAST",
            "status": "COMPLETED",
            "created_at": "2025-04-05T10:30:00Z"
        },
        {
            "id": 2,
            "task_type": "INVENTORY",
            "status": "COMPLETED",
            "created_at": "2025-04-05T09:45:00Z"
        },
        {
            "id": 3,
            "task_type": "PRICING",
            "status": "RUNNING",
            "created_at": "2025-04-05T11:15:00Z"
        }
    ]

def get_product_list():
    """Get list of products"""
    # In a real implementation, this would make an API call
    return [f"P{i:04d}" for i in range(1, 21)]

def get_store_list():
    """Get list of stores"""
    # In a real implementation, this would make an API call
    return [f"S{i:03d}" for i in range(1, 11)]

def create_forecast_task(products, stores, horizon):
    """Create a forecast task"""
    # In a real implementation, this would make an API call
    return {
        "id": 10,
        "task_type": "FORECAST",
        "status": "PENDING",
        "parameters": {
            "action": "forecast",
            "product_ids": products,
            "store_ids": stores,
            "horizon": horizon
        },
        "created_at": "2025-04-05T12:00:00Z"
    }
def show_recent_forecasts():
    """Display recent forecasts"""
    # Create dummy forecast data
    forecast_data = pd.DataFrame({
        "Product ID": ["P0001", "P0002", "P0003", "P0001", "P0002"],
        "Store ID": ["S001", "S001", "S001", "S002", "S002"],
        "Date": pd.date_range(start="2025-04-06", periods=5),
        "Forecast": [120, 85, 50, 95, 65],
        "Confidence": [0.92, 0.88, 0.90, 0.85, 0.87]
    })
    
    st.dataframe(forecast_data)
    
    # Create forecast chart
    fig = px.line(
        forecast_data,
        x="Date",
        y="Forecast",
        color="Product ID",
        facet_col="Store ID",
        title="Demand Forecast by Product and Store",
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_inventory_data(store_filter, status_filter):
    """Get inventory data with filters applied"""
    # Create dummy inventory data
    data = pd.DataFrame({
        "Product ID": ["P0001", "P0002", "P0003", "P0004", "P0005"],
        "Store ID": ["S001", "S001", "S002", "S002", "S003"],
        "Current Stock": [85, 20, 120, 45, 60],
        "Reorder Point": [50, 25, 75, 30, 40],
        "Optimal Stock": [100, 50, 150, 60, 80],
        "Status": ["Optimal", "Low Stock", "Overstocked", "Optimal", "Optimal"],
        "Days to Stockout": [7, 2, 15, 5, 8]
    })
    
    # Apply filters
    if store_filter != "All Stores":
        data = data[data["Store ID"] == store_filter]
    
    if status_filter != "All":
        data = data[data["Status"] == status_filter]
    
    return data

def get_reorder_suggestions():
    """Get reorder suggestions"""
    # Create dummy reorder data
    return pd.DataFrame({
        "Product ID": ["P0002", "P0004", "P0007"],
        "Store ID": ["S001", "S002", "S003"],
        "Current Stock": [20, 18, 12],
        "Forecast Demand (7 days)": [35, 25, 20],
        "Recommended Order": [30, 20, 15],
        "Supplier Lead Time (days)": [3, 2, 4],
        "Urgency": ["High", "Medium", "High"]
    })

def get_pricing_suggestions():
    """Get pricing suggestions"""
    # Create dummy pricing data
    return pd.DataFrame({
        "Product ID": ["P0001", "P0003", "P0005"],
        "Store ID": ["S001", "S002", "S001"],
        "Current Price": [24.99, 15.50, 9.99],
        "Suggested Price": [26.99, 14.75, 10.49],
        "Change (%)": [8.0, -4.8, 5.0],
        "Expected Impact on Demand (%)": [-3.5, 6.2, -2.1],
        "Expected Revenue Change (%)": [4.2, 1.1, 2.8],
        "Elasticity Index": [0.44, 1.29, 0.42]
    })

def create_elasticity_chart(pricing_data):
    """Create price elasticity visualization"""
    # Create dummy elasticity curve
    price_range = np.linspace(0.8, 1.2, 20)
    
    # Different elasticity curves
    low_elasticity = 1 - 0.4 * (price_range - 1)
    high_elasticity = 1 - 1.3 * (price_range - 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_range,
        y=low_elasticity,
        mode='lines',
        name='Low Elasticity Product (0.4)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=price_range,
        y=high_elasticity,
        mode='lines',
        name='High Elasticity Product (1.3)',
        line=dict(color='red')
    ))
    
    # Mark current and suggested prices
    fig.add_trace(go.Scatter(
        x=[1.0],
        y=[1.0],
        mode='markers',
        name='Current Price',
        marker=dict(size=10, color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=[1.08, 0.952, 1.05],
        y=[0.968, 1.062, 0.979],
        mode='markers',
        name='Suggested Price',
        marker=dict(size=10, color='orange')
    ))
    
    fig.update_layout(
        title="Price Elasticity Analysis",
        xaxis_title="Relative Price (Current Price = 1.0)",
        yaxis_title="Relative Demand (Current Demand = 1.0)",
        legend=dict(x=0.01, y=0.99),
        hovermode="closest"
    )
    
    return fig

def display_price_impact_analysis(product, price_change):
    """Display price impact analysis results"""
    # Calculate simulated impact
    elasticity = 0.8  # Dummy value
    demand_impact = -elasticity * price_change
    revenue_impact = (1 + price_change/100) * (1 + demand_impact/100) - 1
    profit_impact = revenue_impact * 1.2  # Assuming higher profit margin impact
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Impact on Demand",
            f"{demand_impact:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Impact on Revenue",
            f"{revenue_impact * 100:.1f}%",
            delta=f"{revenue_impact * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Impact on Profit",
            f"{profit_impact * 100:.1f}%",
            delta=f"{profit_impact * 100:.1f}%"
        )
    
    # Display chart
    st.subheader("Projected Impact over Time")
    dates = pd.date_range(start="2025-04-05", periods=12, freq='D')
    
    baseline = [100] * len(dates)
    new_demand = [100 * (1 + demand_impact/100)] * len(dates)
    new_revenue = [100 * (1 + revenue_impact*100/100)] * len(dates)
    
    impact_df = pd.DataFrame({
        "Date": dates,
        "Baseline": baseline,
        "New Demand": new_demand,
        "New Revenue": new_revenue
    })
    
    fig = px.line(
        impact_df,
        x="Date",
        y=["Baseline", "New Demand", "New Revenue"],
        title=f"Projected Impact of {price_change}% Price Change on {product}"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_pending_actions():
    """Get pending agent actions"""
    # In a real implementation, this would make an API call
    return [
        {
            "id": 1,
            "action_type": "REORDER",
            "status": "SUGGESTED",
            "details": {
                "product_id": "P0002",
                "store_id": "S001",
                "current_stock": 20,
                "forecast_demand": 35,
                "recommended_order": 30
            },
            "reasoning": "Current stock will be depleted in 2 days based on forecast demand. Supplier lead time is 3 days."
        },
        {
            "id": 2,
            "action_type": "PRICE_CHANGE",
            "status": "SUGGESTED",
            "details": {
                "product_id": "P0003",
                "store_id": "S002",
                "current_price": 15.50,
                "suggested_price": 14.75,
                "elasticity": 1.29,
                "projected_demand_increase": "6.2%",
                "projected_revenue_increase": "1.1%"
            },
            "reasoning": "Product has high inventory levels and high price elasticity. Reducing price will increase demand and help reduce excess inventory."
        }
    ]

def approve_action(action_id):
    """Approve an agent action"""
    # In a real implementation, this would make an API call
    pass

def reject_action(action_id, reason):
    """Reject an agent action"""
    # In a real implementation, this would make an API call
    pass

def show_action_history():
    """Display action history"""
    # Create dummy action history data
    history_data = pd.DataFrame({
        "ID": [5, 4, 3],
        "Action Type": ["REORDER", "PRICE_CHANGE", "TRANSFER"],
        "Status": ["EXECUTED", "EXECUTED", "REJECTED"],
        "Created Date": ["2025-04-04", "2025-04-03", "2025-04-03"],
        "Executed Date": ["2025-04-04", "2025-04-03", None],
        "Result": ["Order placed successfully", "Price updated in system", "Insufficient capacity at target location"]
    })
    
    st.dataframe(history_data)

def show_stock_alerts():
    """Display stock alerts"""
    # Create dummy alert data
    alert_data = pd.DataFrame({
        "Product ID": ["P0002", "P0007", "P0010"],
        "Store ID": ["S001", "S003", "S002"],
        "Alert Type": ["Low Stock", "Stockout Risk", "Excess Stock"],
        "Current Stock": [20, 15, 180],
        "Days to Action": [2, 3, 14],
        "Priority": ["High", "Medium", "Low"]
    })
    
    # Color code by priority
    def highlight_priority(val):
        if val == "High":
            return 'background-color: red; color: white'
        elif val == "Medium":
            return 'background-color: orange; color: white'
        elif val == "Low":
            return 'background-color: yellow; color: black'
        else:
            return ''
    
    styled_df = alert_data.style.applymap(highlight_priority, subset=["Priority"])
    st.dataframe(styled_df)

def show_agent_activity():
    """Show agent activity chart"""
    # Create dummy activity data
    dates = pd.date_range(start="2025-03-29", periods=7)
    
    activity_data = pd.DataFrame({
        "Date": dates,
        "Demand Forecasting": [12, 15, 8, 10, 14, 9, 11],
        "Inventory Management": [8, 10, 7, 9, 11, 6, 8],
        "Pricing Optimization": [5, 3, 6, 4, 5, 7, 4],
        "Supply Chain": [3, 4, 2, 5, 3, 2, 4],
        "Customer Behavior": [2, 3, 1, 2, 3, 1, 2]
    })
    
    fig = px.bar(
        activity_data,
        x="Date",
        y=["Demand Forecasting", "Inventory Management", "Pricing Optimization", "Supply Chain", "Customer Behavior"],
        title="Agent Activity by Day",
        labels={"value": "Tasks Completed", "variable": "Agent Type"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_training_metrics(agent_type):
    """Show training metrics for an agent"""
    st.subheader("Training Results")
    
    # Create dummy metrics based on agent type
    if agent_type == "Demand Forecasting Agent":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("RMSE", "6.24", "-1.2")
        
        with col2:
            st.metric("MAE", "4.85", "-0.78")
        
        with col3:
            st.metric("R² Score", "0.87", "+0.03")
        
        # Show feature importance
        st.subheader("Feature Importance")
        feature_data = pd.DataFrame({
            "Feature": ["Price", "Promotions", "Seasonality", "Day of Week", "Month"],
            "Importance": [0.35, 0.28, 0.22, 0.10, 0.05]
        })
        
        fig = px.bar(
            feature_data,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Feature Importance"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif agent_type == "Pricing Optimization Agent":
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Price Prediction Error", "2.4%", "-0.6%")
        
        with col2:
            st.metric("Elasticity Estimation Error", "3.8%", "-1.2%")
        
        # Show elasticity distribution
        st.subheader("Elasticity Distribution by Product Category")
        
        # Dummy data
        elasticity_data = pd.DataFrame({
            "Category": ["Electronics", "Clothing", "Food", "Home", "Beauty"],
            "Mean Elasticity": [0.82, 1.35, 1.50, 0.95, 1.20],
            "Min Elasticity": [0.45, 0.95, 1.10, 0.60, 0.85],
            "Max Elasticity": [1.20, 1.80, 2.10, 1.30, 1.65]
        })
        
        fig = go.Figure()
        
        for i, row in elasticity_data.iterrows():
            fig.add_trace(go.Box(
                name=row["Category"],
                y=[row["Min Elasticity"], row["Mean Elasticity"], row["Max Elasticity"]],
                boxpoints=False
            ))
        
        st.plotly_chart(fig, use_container_width=True)

def show_agent_performance_metrics():
    """Show overall agent performance metrics"""
    st.subheader("Agent Performance Metrics")
    
    # Create dummy performance data
    performance_data = pd.DataFrame({
        "Agent": ["Demand Forecasting", "Inventory Management", "Pricing Optimization", "Supply Chain", "Customer Behavior"],
        "Accuracy": [92.4, 89.7, 85.3, 90.2, 87.6],
        "Tasks Completed": [564, 382, 215, 173, 98],
        "Avg. Processing Time (s)": [0.82, 1.05, 1.32, 0.74, 1.47],
        "Model Version": ["v2.3", "v1.8", "v1.5", "v2.0", "v1.2"],
        "Last Updated": ["2025-04-01", "2025-03-25", "2025-03-18", "2025-03-30", "2025-03-22"]
    })
    
    st.dataframe(performance_data)
    
    # Create performance chart
    st.subheader("Agent Accuracy Over Time")
    
    # Dummy time series data for accuracy
    dates = pd.date_range(start="2025-03-01", periods=30)
    
    np.random.seed(42)  # For reproducibility
    
    accuracy_data = pd.DataFrame({
        "Date": dates,
        "Demand Forecasting": 90 + np.cumsum(np.random.normal(0.1, 0.3, len(dates))),
        "Inventory Management": 85 + np.cumsum(np.random.normal(0.15, 0.25, len(dates))),
        "Pricing Optimization": 80 + np.cumsum(np.random.normal(0.2, 0.3, len(dates))),
        "Supply Chain": 88 + np.cumsum(np.random.normal(0.1, 0.2, len(dates))),
        "Customer Behavior": 82 + np.cumsum(np.random.normal(0.2, 0.25, len(dates)))
    })
    
    # Ensure values stay within reasonable ranges
    for col in accuracy_data.columns:
        if col != "Date":
            accuracy_data[col] = accuracy_data[col].clip(75, 95)
    
    fig = px.line(
        accuracy_data,
        x="Date",
        y=["Demand Forecasting", "Inventory Management", "Pricing Optimization", "Supply Chain", "Customer Behavior"],
        title="Agent Accuracy Trends",
        labels={"value": "Accuracy (%)", "variable": "Agent Type"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()