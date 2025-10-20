import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sde_solver import SDESolver

app = FastAPI(title="SDE Symbolic Solver API", version="1.0.0")


# CORS middleware 
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://sde-solver.netlify.app",  
    "https://*.netlify.app", 
]

# URL Netlify with production
if os.environ.get("NETLIFY_URL"):
    origins.append(os.environ.get("NETLIFY_URL"))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SDEProblem(BaseModel):
    equation_type: str  # "ito" or "stratonovich"
    drift: str
    diffusion: str
    initial_condition: Optional[str] = None
    variables: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, float]] = None

class SimulationRequest(BaseModel):
    equation_type: str
    drift: str
    diffusion: str
    initial_condition: str
    parameters: Dict[str, float]
    time_span: List[float] = [0, 1]
    num_points: int = 100
    num_trajectories: int = 5

class SolutionResponse(BaseModel):
    steps: List[Dict[str, str]]
    final_solution: str
    solution_type: str
    metadata: Dict[str, Any]

class SimulationResponse(BaseModel):
    plot_data: str  # base64 encoded image
    time_points: List[float]
    trajectories: List[List[float]]

@app.get("/")
async def root():
    return {"message": "SDE Symbolic Solver API", "version": "1.0.0"}

@app.post("/solve", response_model=SolutionResponse)
async def solve_sde(problem: SDEProblem):
    try:
        solver = SDESolver()
        
        # Parse the problem
        result = solver.solve(
            equation_type=problem.equation_type,
            drift=problem.drift,
            diffusion=problem.diffusion,
            initial_condition=problem.initial_condition,
            variables=problem.variables or {},
            parameters=problem.parameters or {}
        )
        
        return SolutionResponse(
            steps=result["steps"],
            final_solution=result["final_solution"],
            solution_type=result["solution_type"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error solving SDE: {str(e)}")

@app.post("/simulate")
async def simulate_sde(sim_request: SimulationRequest):
    try:
        # Generate simulation data
        plot_data, time_points, trajectories = generate_simulation_plot(sim_request)
        
        return JSONResponse(content={
            "plot_data": plot_data,
            "time_points": time_points,
            "trajectories": trajectories
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error simulating SDE: {str(e)}")

def generate_simulation_plot(sim_request: SimulationRequest):
    """Generate simulation plot for SDE"""
    # Set Palatino font
    plt.rcParams.update({
        'font.family': 'Palatino',
        'font.serif': 'Palatino',
        'mathtext.fontset': 'cm',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Create figure with specific style
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Generate sample data (in practice, this would use actual SDE simulation)
    t = np.linspace(sim_request.time_span[0], sim_request.time_span[1], sim_request.num_points)
    
    trajectories = []
    for i in range(sim_request.num_trajectories):
        # Simple geometric Brownian motion simulation as example
        mu = sim_request.parameters.get('mu', 0.1)
        sigma = sim_request.parameters.get('sigma', 0.2)
        x0 = float(sim_request.initial_condition) if sim_request.initial_condition else 1.0
        
        # Euler-Maruyama simulation
        dt = t[1] - t[0]
        x = np.zeros_like(t)
        x[0] = x0
        
        for j in range(1, len(t)):
            dW = np.random.normal(0, np.sqrt(dt))
            x[j] = x[j-1] + mu * x[j-1] * dt + sigma * x[j-1] * dW
        
        trajectories.append(x.tolist())
        ax.plot(t, x, color='black', linewidth=1.2, alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Time (t)', fontsize=12, fontname='Palatino')
    ax.set_ylabel('X(t)', fontsize=12, fontname='Palatino')
    ax.set_title('SDE Simulation', fontsize=14, fontname='Palatino', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add signature
    ax.text(0.02, 0.98, 'SDE Solver by Ramanambonona Ambinintsoa, PhD', 
            transform=ax.transAxes, fontsize=10, fontname='Palatino',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_data, t.tolist(), trajectories

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Additional endpoints for examples
@app.get("/examples")
async def get_examples():
    examples = {
        "geometric_brownian": {
            "name": "Geometric Brownian Motion",
            "drift": "mu*x",
            "diffusion": "sigma*x",
            "description": "Used in financial mathematics for stock prices",
            "parameters": {"mu": 0.1, "sigma": 0.2}
        },
        "ornstein_uhlenbeck": {
            "name": "Ornstein-Uhlenbeck Process",
            "drift": "theta*(mu - x)",
            "diffusion": "sigma",
            "description": "Mean-reverting process used in interest rates modeling",
            "parameters": {"theta": 1.0, "mu": 0.0, "sigma": 0.5}
        },
        "vasicek_model": {
            "name": "Vasicek Model",
            "drift": "a*(b - x)",
            "diffusion": "sigma",
            "description": "Interest rate model with mean reversion",
            "parameters": {"a": 0.1, "b": 0.05, "sigma": 0.02}
        }
    }
    return examples

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
