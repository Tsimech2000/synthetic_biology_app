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

def stochastic_toggle_switch():
    st.header("Stochastic Genetic Toggle Switch")
    st.write("This section simulates a toggle switch with stochastic effects using the Gillespie algorithm.")
    
    # Sidebar inputs
    alpha = st.sidebar.slider("Transcription Rate (α)", 0.1, 20.0, 10.0)
    beta = st.sidebar.slider("Cooperativity (β)", 1.0, 5.0, 2.0)
    gamma = st.sidebar.slider("Degradation Rate (γ)", 0.01, 1.0, 0.1)
    t_max = st.sidebar.slider("Simulation Time", 10, 500, 100)
    
    # Gillespie Algorithm Implementation
    def gillespie_toggle(alpha, beta, gamma, max_time, max_steps=5000):
        time = 0
        u, v = 1, 1
        time_points = [time]
        u_levels, v_levels = [u], [v]
        
        for _ in range(max_steps):  # Prevent infinite loops
            if time >= max_time:
                break
            
            prod_u = alpha / (1 + v**beta)
            prod_v = alpha / (1 + u**beta)
            deg_u = gamma * u
            deg_v = gamma * v
            
            total_rate = prod_u + prod_v + deg_u + deg_v
            if total_rate == 0:
                break
            
            time += -np.log(random.random()) / total_rate
            event = random.choices(['prod_u', 'prod_v', 'deg_u', 'deg_v'], weights=[prod_u, prod_v, deg_u, deg_v])[0]
            
            if event == 'prod_u':
                u += 1
            elif event == 'prod_v':
                v += 1
            elif event == 'deg_u' and u > 0:
                u -= 1
            elif event == 'deg_v' and v > 0:
                v -= 1
            
            time_points.append(time)
            u_levels.append(u)
            v_levels.append(v)
        
        return time_points, u_levels, v_levels
    
    # Run the simulation
    time_points, u_levels, v_levels = gillespie_toggle(alpha, beta, gamma, t_max)
    
    # Ensure time points are sorted for Plotly
    sorted_indices = np.argsort(time_points)
    time_points = np.array(time_points)[sorted_indices]
    u_levels = np.array(u_levels)[sorted_indices]
    v_levels = np.array(v_levels)[sorted_indices]
    
    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_points, y=u_levels, mode='lines', name='Gene A Expression'))
    fig.add_trace(go.Scatter(x=time_points, y=v_levels, mode='lines', name='Gene B Expression'))
    fig.update_layout(title="Stochastic Toggle Switch Dynamics", xaxis_title="Time", yaxis_title="Expression Level")
    
    st.plotly_chart(fig)

# Streamlit App
st.title("Synthetic Biology Multi-Simulation App")

# Sidebar for navigation
simulation_choice = st.sidebar.selectbox(
    "Choose a Simulation:",
    ("Genetic Oscillator (Repressilator)", 
     "Stochastic Genetic Toggle Switch")
)

# Load and run the selected simulation
dispatcher = {
    "Genetic Oscillator (Repressilator)": genetic_oscillator,
    "Stochastic Genetic Toggle Switch": stochastic_toggle_switch,
}

dispatcher[simulation_choice]()

