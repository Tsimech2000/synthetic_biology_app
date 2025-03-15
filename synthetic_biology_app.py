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

def crispr_logic_gates():
    st.header("CRISPR-Based Logic Gate Simulator")
    st.write("Simulate CRISPR-based genetic logic gates (NOT, AND, OR).")
    
    # User selects logic gate type
    gate_type = st.sidebar.selectbox("Select Logic Gate:", ["NOT", "AND", "OR"])
    
    # User inputs for gRNA presence
    gRNA1 = st.sidebar.checkbox("gRNA 1 Present")
    gRNA2 = st.sidebar.checkbox("gRNA 2 Present")
    
    # Logic Gate Implementation
    if gate_type == "NOT":
        output = 0 if gRNA1 else 1  # NOT gate: gene OFF if gRNA present
    elif gate_type == "AND":
        output = 0 if (gRNA1 and gRNA2) else 1  # AND gate: gene OFF if both gRNAs present
    elif gate_type == "OR":
        output = 0 if (gRNA1 or gRNA2) else 1  # OR gate: gene OFF if at least one gRNA present
    
    # Display output
    st.subheader(f"Gene Expression Output: {output}")

def stochastic_gene_expression():
    st.header("Stochastic Gene Expression Simulator")
    st.write("This section simulates stochastic gene expression using the Gillespie algorithm.")
    
    # Sidebar inputs for reaction rates
    transcription_rate = st.sidebar.slider("Transcription Rate", 0.1, 5.0, 1.0)
    degradation_rate = st.sidebar.slider("Degradation Rate", 0.01, 1.0, 0.1)
    max_time = st.sidebar.slider("Simulation Time", 10, 500, 100)
    
    # Gillespie Algorithm Implementation
    def gillespie_simulation(transcription_rate, degradation_rate, max_time):
        time = 0
        mRNA_count = 0
        time_points = [time]
        mRNA_levels = [mRNA_count]
        
        while time < max_time:
            transcription_prob = transcription_rate
            degradation_prob = degradation_rate * mRNA_count
            total_rate = transcription_prob + degradation_prob
            
            if total_rate == 0:
                break
            
            time += -np.log(random.random()) / total_rate
            
            if random.random() < transcription_prob / total_rate:
                mRNA_count += 1
            else:
                if mRNA_count > 0:
                    mRNA_count -= 1
            
            time_points.append(time)
            mRNA_levels.append(mRNA_count)
        
        return time_points, mRNA_levels
    
    # Run the simulation
    time_points, mRNA_levels = gillespie_simulation(transcription_rate, degradation_rate, max_time)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(time_points, mRNA_levels, where='post')
    ax.set_xlabel('Time')
    ax.set_ylabel('mRNA Count')
    ax.set_title('Stochastic Gene Expression (Gillespie Algorithm)')
    ax.grid(True)
    
    st.pyplot(fig)

def quorum_sensing():
    st.header("Bacterial Quorum Sensing Simulator")
    st.write("This section models bacterial communication through quorum sensing.")
    
    # Sidebar inputs
    population = st.sidebar.slider("Initial Bacterial Population", 1, 1000, 100)
    production_rate = st.sidebar.slider("Autoinducer Production Rate", 0.1, 5.0, 1.0)
    degradation_rate = st.sidebar.slider("Autoinducer Degradation Rate", 0.01, 1.0, 0.1)
    threshold = st.sidebar.slider("Activation Threshold", 10, 500, 100)
    
    time = np.linspace(0, 50, 500)
    autoinducer = production_rate * population * (1 - np.exp(-degradation_rate * time))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, autoinducer, label='Autoinducer Level')
    ax.axhline(threshold, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Autoinducer Level')
    ax.set_title('Bacterial Quorum Sensing Activation')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

# Streamlit App
st.title("Synthetic Biology Multi-Simulation App")

# Sidebar for navigation
simulation_choice = st.sidebar.selectbox(
    "Choose a Simulation:",
    ("Genetic Oscillator (Repressilator)", 
     "CRISPR-Based Logic Gate Simulator", 
     "Stochastic Gene Expression Simulator", 
     "Bacterial Quorum Sensing Simulator")
)

# Load and run the selected simulation
dispatcher = {
    "Genetic Oscillator (Repressilator)": genetic_oscillator,
    "CRISPR-Based Logic Gate Simulator": crispr_logic_gates,
    "Stochastic Gene Expression Simulator": stochastic_gene_expression,
    "Bacterial Quorum Sensing Simulator": quorum_sensing,
}

dispatcher[simulation_choice]()
