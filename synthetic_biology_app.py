import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import random

def genetic_oscillator():
    st.header("Genetic Oscillator Simulator (Repressilator)")
    st.write("This section simulates the Repressilator model, a synthetic genetic oscillator.")
    
    # Sidebar inputs
    alpha = st.sidebar.slider("Transcription Rate (α)", 1.0, 20.0, 10.0)
    beta = st.sidebar.slider("Cooperativity (β)", 1.0, 5.0, 2.0)
    gamma = st.sidebar.slider("Degradation Rate (γ)", 0.1, 5.0, 1.0)
    t_max = st.sidebar.slider("Simulation Time", 10, 200, 100)
    
    # Define ODEs for the Repressilator model
    def repressilator(y, t, alpha, beta, gamma):
        m1, m2, m3, p1, p2, p3 = y
        dm1_dt = alpha / (1 + p3**beta) - m1
        dm2_dt = alpha / (1 + p1**beta) - m2
        dm3_dt = alpha / (1 + p2**beta) - m3
        dp1_dt = m1 - gamma * p1
        dp2_dt = m2 - gamma * p2
        dp3_dt = m3 - gamma * p3
        return [dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt]
    
    # Initial conditions
    y0 = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    t = np.linspace(0, t_max, 1000)
    
    # Solve ODEs
    sol = odeint(repressilator, y0, t, args=(alpha, beta, gamma))
    
    # Extract solutions
    m1, m2, m3, p1, p2, p3 = sol.T
    
    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=p1, mode='lines', name='Protein 1'))
    fig.add_trace(go.Scatter(x=t, y=p2, mode='lines', name='Protein 2'))
    fig.add_trace(go.Scatter(x=t, y=p3, mode='lines', name='Protein 3'))
    fig.update_layout(title="Repressilator Dynamics", xaxis_title="Time", yaxis_title="Protein Concentration")
    
    st.plotly_chart(fig)

def enzyme_kinetics():
    st.header("Enzyme Kinetics Simulator")
    st.write("This section simulates enzyme kinetics using Michaelis-Menten equations.")
    
    # Sidebar inputs
    Vmax = st.sidebar.slider("Maximum Reaction Rate (Vmax)", 0.1, 10.0, 1.0)
    Km = st.sidebar.slider("Michaelis Constant (Km)", 0.1, 10.0, 1.0)
    S0 = st.sidebar.slider("Initial Substrate Concentration (S0)", 0.1, 50.0, 10.0)
    t_max = st.sidebar.slider("Simulation Time", 10, 200, 100)
    
    # Define Michaelis-Menten equation
    def michaelis_menten(y, t, Vmax, Km):
        S, P = y
        dS_dt = -Vmax * S / (Km + S)
        dP_dt = Vmax * S / (Km + S)
        return [dS_dt, dP_dt]
    
    # Initial conditions
    y0 = [S0, 0.0]
    t = np.linspace(0, t_max, 500)
    
    # Solve ODEs
    sol = odeint(michaelis_menten, y0, t, args=(Vmax, Km))
    
    # Extract solutions
    S, P = sol.T
    
    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Substrate'))
    fig.add_trace(go.Scatter(x=t, y=P, mode='lines', name='Product'))
    fig.update_layout(title="Enzyme Kinetics Simulation", xaxis_title="Time", yaxis_title="Concentration")
    
    st.plotly_chart(fig)

# Streamlit App
st.title("Synthetic Biology Multi-Simulation App")

# Sidebar for navigation
simulation_choice = st.sidebar.selectbox(
    "Choose a Simulation:",
    ("Genetic Oscillator (Repressilator)", 
     "Stochastic Genetic Toggle Switch Simulator",
     "Enzyme Kinetics Simulator")
)

# Load and run the selected simulation
dispatcher = {
    "Genetic Oscillator (Repressilator)": genetic_oscillator,
    "Stochastic Genetic Toggle Switch Simulator": stochastic_toggle_switch,
    "Enzyme Kinetics Simulator": enzyme_kinetics,
}

dispatcher[simulation_choice]()

