import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
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
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, p1, label='Protein 1')
    ax.plot(t, p2, label='Protein 2')
    ax.plot(t, p3, label='Protein 3')
    ax.set_xlabel('Time')
    ax.set_ylabel('Protein Concentration')
    ax.set_title('Repressilator Dynamics')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

def stochastic_toggle_switch():
    st.header("Stochastic Genetic Toggle Switch")
    st.write("Simulating a genetic toggle switch with stochastic noise using the Gillespie Algorithm.")
    
    # Sidebar inputs
    alpha1 = st.sidebar.slider("Alpha 1 (Gene A)", 0.1, 20.0, 10.0)
    alpha2 = st.sidebar.slider("Alpha 2 (Gene B)", 0.1, 20.0, 10.0)
    beta = st.sidebar.slider("Beta (Repression Strength)", 1.0, 5.0, 2.0)
    gamma = st.sidebar.slider("Degradation Rate", 0.01, 1.0, 0.1)
    t_max = st.sidebar.slider("Simulation Time", 10, 500, 100)
    
    # Gillespie Algorithm for stochastic simulation
    def gillespie_toggle(alpha1, alpha2, beta, gamma, max_time):
        time = 0
        u, v = 1, 1
        time_points = [time]
        u_levels, v_levels = [u], [v]
        
        while time < max_time:
            prod_u = alpha1 / (1 + v**beta)
            prod_v = alpha2 / (1 + u**beta)
            deg_u = gamma * u
            deg_v = gamma * v
            
            total_rate = prod_u + prod_v + deg_u + deg_v
            
            if total_rate == 0:
                break
            
            time += -np.log(random.random()) / total_rate
            event = random.choices(['prod_u', 'prod_v', 'deg_u', 'deg_v'], 
                                   weights=[prod_u, prod_v, deg_u, deg_v])[0]
            
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
    
    # Run stochastic simulation
    time_points, u_levels, v_levels = gillespie_toggle(alpha1, alpha2, beta, gamma, t_max)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(time_points, u_levels, where='post', label='Protein U (Gene A)')
    ax.step(time_points, v_levels, where='post', label='Protein V (Gene B)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Protein Count')
    ax.set_title('Stochastic Toggle Switch Simulation')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

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
