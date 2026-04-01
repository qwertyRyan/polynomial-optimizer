import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

# --- Helper Functions ---
def poly_eval(vars, coeffs):
    x1, x2 = vars
    a0, a1, a2, a12, a11, a22, a112, a122, a1122, a111, a222 = coeffs
    return (a0 + a1*x1 + a2*x2 + a12*x1*x2 + a11*x1**2 + a22*x2**2 + 
            a112*x1*x2**2 + a122*x1**2*x2 + a1122*x1**2*x2**2 + 
            a111*x1**3 + a222*x2**3)

st.set_page_config(layout="wide")
st.title("Polynomial Optimizer Pro")

# --- UI Setup ---
cols_header = ["a0", "a1", "a2", "a12", "a11", "a22", "a112", "a122", "a1122", "a111", "a222"]

with st.sidebar:
    st.header("Solver Settings")
    mode = st.selectbox("Mode", ["min", "max"])
    num_runs = st.number_input("Number of Optimization Runs", min_value=1, value=50)
    
    st.subheader("Bounds")
    c1, c2 = st.columns(2)
    x1_min = c1.number_input("x1 Min", value=-1.0, format="%.8f")
    x1_max = c2.number_input("x1 Max", value=1.0, format="%.8f")
    
    c3, c4 = st.columns(2)
    x2_min = c3.number_input("x2 Min", value=-1.0, format="%.8f")
    x2_max = c4.number_input("x2 Max", value=1.0, format="%.8f")
    
    bounds = [(x1_min, x1_max), (x2_min, x2_max)]

# --- Input Sections ---
st.subheader("Objective Function Coefficients")
df_obj = st.data_editor(pd.DataFrame([[0.0]*11], columns=cols_header), num_rows="fixed")
obj_coeffs = df_obj.iloc[0].tolist()

st.subheader("Constraints")
n = st.number_input("Number of Constraints", min_value=1, value=1)
constraints_data = []

for i in range(n):
    with st.expander(f"Constraint {i+1}"):
        df_c = st.data_editor(pd.DataFrame([[0.0]*11], columns=cols_header), key=f"c_{i}", num_rows="fixed")
        c1, c2 = st.columns(2)
        lb = c1.number_input("Lower Bound", value=0.0, key=f"lb_{i}", format="%.8f")
        ub = c2.number_input("Upper Bound", value=10.0, key=f"ub_{i}", format="%.8f")
        constraints_data.append((df_c.iloc[0].tolist(), lb, ub))

# --- Execution ---
if st.button("Run Optimization"):
    best_overall_res = None
    min_penalty_val = float('inf')
    progress_bar = st.progress(0)

    for i in range(num_runs):
        def combined_objective(vars):
            obj_val = poly_eval(vars, obj_coeffs)
            if mode == 'max': obj_val *= -1
            
            penalty = 0
            for coeffs, lb, ub in constraints_data:
                val = poly_eval(vars, coeffs)
                if val < lb: penalty += 1e7 * (lb - val)**2
                elif val > ub: penalty += 1e7 * (val - ub)**2
            return obj_val + penalty

        res = differential_evolution(combined_objective, bounds)
        if res.fun < min_penalty_val:
            min_penalty_val = res.fun
            best_overall_res = res
        progress_bar.progress((i + 1) / num_runs)

    # --- Results Display ---
    st.divider()
    st.success("Optimization Complete!")
    
    final_x = best_overall_res.x
    raw_objective = poly_eval(final_x, obj_coeffs)
    
    applied_penalty = 0
    for coeffs, lb, ub in constraints_data:
        val = poly_eval(final_x, coeffs)
        if val < lb: applied_penalty += 1e7 * (lb - val)**2
        elif val > ub: applied_penalty += 1e7 * (val - ub)**2

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Optimal x1", f"{final_x[0]:.8f}")
    col_r2.metric("Optimal x2", f"{final_x[1]:.8f}")
    col_r3.metric("Raw Objective f(x)", f"{raw_objective:.8f}")
    
    st.info(f"**Penalty Value at Result:** {applied_penalty:.8f}")
    
    st.subheader("Constraint Analysis")
    for i, (coeffs, lb, ub) in enumerate(constraints_data):
        val = poly_eval(final_x, coeffs)
        is_met = (lb <= val <= ub)
        status = "MET" if is_met else "VIOLATED"
        color = "green" if is_met else "red"
        
        # Explicitly show values for failed constraints
        violation_msg = ""
        if not is_met:
            if val < lb:
                violation_msg = f" (Value: {val:.8f} is below Lower Bound by {(lb-val):.8f})"
            else:
                violation_msg = f" (Value: {val:.8f} is above Upper Bound by {(val-ub):.8f})"
            
        st.markdown(f":{color}[**Constraint {i+1}**: {status} | Target: {lb:.8f} to {ub:.8f}{violation_msg}]")