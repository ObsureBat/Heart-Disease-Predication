"""Visualization components"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import numpy as np

def create_risk_display(probability):
    """Create HTML for risk assessment display"""
    color = '#ff4b4b' if probability > 0.5 else '#4bb543'
    risk_level = 'High' if probability > 0.5 else 'Low'
    
    return f"""
        <div style="padding: 1.5rem; 
                    background: {color}; 
                    border-radius: 0.75rem; 
                    color: white; 
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="font-size: 2rem; margin: 0 0 0.5rem 0;">{probability:.1%}</h2>
            <p style="font-size: 1.25rem; margin: 0;">{risk_level} Risk of Heart Disease</p>
        </div>
    """

def create_feature_importance_plot(model, input_scaled, feature_names):
    """Create feature importance plot using SHAP values"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create feature importance bar chart
    importance_values = np.abs(shap_values[0])
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=importance_df['Feature'],
        x=importance_df['Importance'],
        orientation='h',
        marker_color='#ff4b4b'
    ))
    
    fig.update_layout(
        title="Feature Importance Analysis",
        height=400,
        margin=dict(l=150, r=20, t=40, b=20),
        xaxis_title="Impact on Prediction",
        yaxis_title="",
        template="plotly_white"
    )
    
    return fig