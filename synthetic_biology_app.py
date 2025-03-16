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

def crispr_logic_gate():
    st.header("CRISPR-Based Logic Gate Simulator")
    st.write("This section simulates gene regulation using CRISPR logic gates.")
    
    # Sidebar inputs
    k1 = st.sidebar.slider("Activation Rate (k1)", 0.1, 5.0, 1.0)
    k2 = st.sidebar.slider("Repression Rate (k2)", 0.1, 5.0, 1.0)
    t_max = st.sidebar.slider("Simulation Time", 10, 200, 100)
    
    # Define CRISPR logic gate model
    def crispr_model(y, t, k1, k2):
        A, B = y
        dA_dt = k1 - k2 * A
        dB_dt = k1 * A - k2 * B
        return [dA_dt, dB_dt]
    
    # Initial conditions
    y0 = [0.0, 0.0]
    t = np.linspace(0, t_max, 500)
    
    # Solve ODEs
    sol = odeint(crispr_model, y0, t, args=(k1, k2))
    
    # Extract solutions
    A, B = sol.T
    
    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=A, mode='lines', name='Gene A Expression'))
    fig.add_trace(go.Scatter(x=t, y=B, mode='lines', name='Gene B Expression'))
    fig.update_layout(title="CRISPR Logic Gate Simulation", xaxis_title="Time", yaxis_title="Expression Level")
    
    st.plotly_chart(fig)

def quorum_sensing():
    st.header("Bacterial Quorum Sensing Simulator")
    st.write("Simulating bacterial communication via autoinducer accumulation.")
    
    # Sidebar inputs
    r = st.sidebar.slider("Autoinducer Production Rate (r)", 0.1, 5.0, 1.0)
    d = st.sidebar.slider("Degradation Rate (d)", 0.01, 1.0, 0.1)
    t_max = st.sidebar.slider("Simulation Time", 10, 200, 100)
    
    # Define quorum sensing model
    def quorum_model(y, t, r, d):
        A = y[0]
        dA_dt = r * A - d * A
        return [dA_dt]
    
    # Initial conditions
    y0 = [1.0]
    t = np.linspace(0, t_max, 500)
    
    # Solve ODEs
    sol = odeint(quorum_model, y0, t, args=(r, d))
    
    # Extract solutions
    A = sol.T[0]
    
    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=A, mode='lines', name='Autoinducer Concentration'))
    fig.update_layout(title="Bacterial Quorum Sensing Simulation", xaxis_title="Time", yaxis_title="Concentration")
    
    st.plotly_chart(fig)

# Streamlit App
st.title("Synthetic Biology Multi-Simulation App")

# Sidebar for navigation
simulation_choice = st.sidebar.selectbox(
    "Choose a Simulation:",
    ("Genetic Oscillator (Repressilator)", 
     "CRISPR-Based Logic Gate Simulator",
     "Bacterial Quorum Sensing Simulator",
     "Enzyme Kinetics Simulator")
)

# Load and run the selected simulation
dispatcher = {
    "Genetic Oscillator (Repressilator)": genetic_oscillator,
    "CRISPR-Based Logic Gate Simulator": crispr_logic_gate,
    "Bacterial Quorum Sensing Simulator": quorum_sensing,
    "Enzyme Kinetics Simulator": enzyme_kinetics,
}

dispatcher[simulation_choice]()
