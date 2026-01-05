import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==========================================
# LAYER 1: DATA ACCESS LAYER
# ==========================================
# Load artifacts (Ensure these paths match your folder structure)
try:
    SCALER = joblib.load('models/churn_scaler.joblib')
    FEATURE_NAMES = joblib.load('models/feature_names.joblib')
    MODELS = {
        'lr': joblib.load('models/logistic_regression_model.joblib'),
        'rf': joblib.load('models/random_forest_model.joblib'),
        'xgb': joblib.load('models/xgboost_model.joblib')
    }
except Exception as e:
    print(f"Error loading models: {e}")

# ==========================================
# LAYER 2: BUSINESS LOGIC LAYER
# ==========================================
def calculate_churn_probability(model_key, tenure, monthly, contract, internet, payment):
    input_df = pd.DataFrame(0.0, index=[0], columns=FEATURE_NAMES)

    # Numerical Mapping
    input_df['tenure'] = float(tenure)
    input_df['MonthlyCharges'] = float(monthly)
    input_df['TotalCharges'] = float(tenure * monthly)
    input_df['AvgMonthlySpend'] = float(monthly) if tenure == 0 else float((tenure * monthly) / tenure)

    if 'NumServices' in input_df.columns:
        input_df['NumServices'] = 3.0

    # Categorical Mapping
    for prefix, val in [('Contract', contract), ('InternetService', internet), ('PaymentMethod', payment)]:
        col_name = f"{prefix}_{val}"
        if col_name in input_df.columns:
            input_df[col_name] = 1.0

    # Scale only numeric columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumServices', 'AvgMonthlySpend']
    num_cols = [c for c in num_cols if c in FEATURE_NAMES]
    input_df[num_cols] = SCALER.transform(input_df[num_cols])

    return MODELS[model_key].predict_proba(input_df)[0][1]

def get_explanation_plot(model_key):
    model = MODELS[model_key]
    importances = np.abs(model.coef_[0]) if model_key == 'lr' else model.feature_importances_

    df_imp = pd.Series(importances, index=FEATURE_NAMES).nlargest(10).reset_index()
    df_imp.columns = ['Feature', 'Importance']
    df_imp = df_imp.sort_values(by='Importance', ascending=True)

    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h',
                 title=f"Key Drivers ({model_key.upper()})",
                 color='Importance', color_continuous_scale='Blues',
                 template='simple_white', text_auto='.2f')

    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=40, b=10),
                      coloraxis_showscale=False, yaxis={'title': ''}, xaxis={'title': 'Impact Score'})
    return fig

# ==========================================
# LAYER 3: PRESENTATION LAYER
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server  # CRITICAL FOR RENDER DEPLOYMENT

def get_valid_options(prefix):
    return [{'label': col.replace(f"{prefix}_", ""), 'value': col.replace(f"{prefix}_", "")}
            for col in FEATURE_NAMES if col.startswith(f"{prefix}_")]

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Customer Retention Intelligence", className="text-center my-4 text-primary"))),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Customer Configuration"),
                dbc.CardBody([
                    html.Label("Model Selection"),
                    dcc.Dropdown(id='model-selector', options=[{'label': k.upper(), 'value': k} for k in MODELS.keys()], value='xgb', className="mb-3"),
                    html.Label("Tenure (Months)"),
                    dcc.Slider(id='tenure-slider', min=1, max=72, value=12, className="mb-4"),
                    html.Label("Monthly Charges ($)"),
                    dbc.Input(id='monthly-input', type='number', value=70, className="mb-3"),
                    html.Label("Contract Type"),
                    dcc.Dropdown(id='contract-dropdown', options=get_valid_options('Contract'), value=get_valid_options('Contract')[0]['value'], className="mb-3"),
                    html.Label("Internet Service"),
                    dcc.Dropdown(id='internet-dropdown', options=get_valid_options('InternetService'), value=get_valid_options('InternetService')[0]['value'], className="mb-3"),
                    html.Label("Payment Method"),
                    dcc.Dropdown(id='payment-dropdown', options=get_valid_options('PaymentMethod'), value=get_valid_options('PaymentMethod')[0]['value']),
                ])
            ], className="shadow border-0")
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk & Analysis Insights"),
                dbc.CardBody([
                    html.Div(id='prediction-result', className="text-center mb-2"),
                    html.Hr(className="my-3"),
                    dcc.Graph(id='importance-graph', config={'responsive': True, 'displayModeBar': False})
                ], style={"padding": "20px"})
            ], className="shadow border-0", style={"height": "100%", "maxHeight": "580px"})
        ], width=8)
    ])
], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# ==========================================
# LAYER 4: CONTROLLER LAYER
# ==========================================
@app.callback(
    [Output('prediction-result', 'children'), Output('importance-graph', 'figure')],
    [Input('model-selector', 'value'), Input('tenure-slider', 'value'),
     Input('monthly-input', 'value'), Input('contract-dropdown', 'value'),
     Input('internet-dropdown', 'value'), Input('payment-dropdown', 'value')]
)
def update_dashboard(model_key, tenure, monthly, contract, internet, payment):
    prob = calculate_churn_probability(model_key, tenure, monthly, contract, internet, payment)
    fig = get_explanation_plot(model_key)

    color = "danger" if prob > 0.5 else "success"
    result_ui = html.Div([
        html.H1(f"{prob:.1%}", className=f"text-{color} display-2 font-weight-bold"),
        dbc.Progress(value=prob*100, color=color, style={"height": "10px"}, className="mb-2"),
        html.H5("Churn Probability Score", className="text-muted")
    ])
    return result_ui, fig

if __name__ == '__main__':
    app.run_server(debug=True)
