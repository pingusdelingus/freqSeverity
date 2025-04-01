from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

app = Flask(__name__)

def run_poisson_simulation(lambda_rate=0.1, mu=10, sigma=0.5, T=30, num_simulations=10000, portfolio_value=100000):
    total_losses = []
    
    for _ in range(num_simulations):
        num_hacks = np.random.poisson(lambda_rate * T)
        
        if num_hacks > 0:
            losses = np.random.lognormal(mean=mu, sigma=sigma, size=num_hacks)
            # Cap losses at portfolio value
            losses = np.minimum(losses, portfolio_value)
        else:
            losses = []
            
        total_loss = sum(losses)
        total_losses.append(total_loss)
    
    return total_losses

def run_stochastic_simulation(lambda_rate=0.1, mu=10, sigma=0.5, T=30, num_simulations=10000, portfolio_value=100000):
    total_losses = []
    time_steps = np.linspace(0, T, 100)
    
    for _ in range(num_simulations):
        # Simulate intensity process (Cox process)
        intensity = lambda_rate * np.exp(np.random.normal(0, 0.1, len(time_steps)))
        
        # Simulate number of events using inhomogeneous Poisson process
        num_hacks = np.random.poisson(np.mean(intensity) * T)
        
        if num_hacks > 0:
            # Add time dependency to severity
            time_factor = np.random.normal(1, 0.1)  # Random time scaling factor
            losses = np.random.lognormal(mean=mu, sigma=sigma, size=num_hacks) * time_factor
            # Cap losses at portfolio value
            losses = np.minimum(losses, portfolio_value)
        else:
            losses = []
            
        total_loss = sum(losses)
        total_losses.append(total_loss)
    
    return total_losses

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.get_json()
    model_type = data.get('model_type', 'poisson')
    
    # Get parameters from request or use defaults
    lambda_rate = float(data.get('lambda_rate', 0.1))
    mu = float(data.get('mu', 10))
    sigma = float(data.get('sigma', 0.5))
    T = int(data.get('T', 30))
    num_simulations = int(data.get('num_simulations', 10000))
    portfolio_value = float(data.get('portfolio_value', 100000))
    
    if model_type == 'poisson':
        total_losses = run_poisson_simulation(lambda_rate, mu, sigma, T, num_simulations, portfolio_value)
    else:
        total_losses = run_stochastic_simulation(lambda_rate, mu, sigma, T, num_simulations, portfolio_value)
    
    # Calculate statistics
    mean_loss = np.mean(total_losses)
    var_95 = np.percentile(total_losses, 95)
    cvar_95 = np.mean([loss for loss in total_losses if loss >= var_95])
    
    # Create histogram using plotly
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=total_losses,
        name='Loss Distribution',
        nbinsx=50,
        opacity=0.7,
        marker_color='#89b4fa'  # Catppuccin blue
    ))
    
    fig.add_vline(x=var_95, line_dash="dash", line_color="#f38ba8", name="95% VaR")  # Catppuccin red
    fig.add_vline(x=mean_loss, line_dash="dash", line_color="#a6e3a1", name="Expected Loss")  # Catppuccin green
    
    fig.update_layout(
        title="Simulation Results for Web3 Wallet Hacking",
        xaxis_title="Loss ($)",
        yaxis_title="Frequency",
        showlegend=True,
        height=600,
        paper_bgcolor='#1e1e2e',  # Catppuccin base
        plot_bgcolor='#313244',    # Catppuccin surface0
        font=dict(color='#cdd6f4')  # Catppuccin text
    )
    
    return jsonify({
        'plot': fig.to_json(),
        'stats': {
            'mean_loss': float(mean_loss),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95)
        }
    })

if __name__ == '__main__':
    app.run(debug=True) 