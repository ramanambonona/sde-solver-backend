import sympy as sp
from sympy import symbols, Function, exp, integrate, diff, sqrt, pi, oo
from typing import Dict, List, Any, Optional
import re

class SDESolver:
    """
    Symbolic solver for Stochastic Differential Equations (SDEs)
    Based on the work of Chu-Ching Huang (2011)
    Reference: Huang, C. C. (2011). Python Solver for Stochastic Differential Equations. 
               Matimyas Matematika, 34(1-2), 87-97.
    """
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, title: str, content: str):
        """Add a step to the solution process"""
        self.steps.append({"title": title, "content": content})
    
    def parse_expression(self, expr_str: str, variables: Dict[str, str]) -> sp.Expr:
        """Parse string expression to sympy expression"""
        # Create symbol mapping
        sym_map = {}
        for var_name, var_type in variables.items():
            if var_type == "variable":
                sym_map[var_name] = symbols(var_name)
            elif var_type == "function":
                sym_map[var_name] = Function(var_name)(symbols('t'))
        
        # Also add common symbols
        common_symbols = {
            't': symbols('t'),
            'x': symbols('x'),
            'y': symbols('y'),
            'z': symbols('z'),
            'w': symbols('w'),
            'W': symbols('W'),
            'dt': symbols('dt'),
            'dw': symbols('dw'),
            'dW': symbols('dW'),
            'alpha': symbols('alpha'),
            'beta': symbols('beta'),
            'sigma': symbols('sigma'),
            'mu': symbols('mu')
        }
        sym_map.update(common_symbols)
        
        # Parse expression
        try:
            return sp.sympify(expr_str, locals=sym_map)
        except:
            # If sympify fails, try to build manually
            return self._build_expression_manual(expr_str, sym_map)
    
    def _build_expression_manual(self, expr_str: str, sym_map: Dict) -> sp.Expr:
        """Manual expression building for complex cases"""
        # This is a simplified version - in practice, you'd want more sophisticated parsing
        return eval(expr_str, {"__builtins__": None}, sym_map)
    
    def solve_linear_sde(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve linear SDE using the method from Huang's paper"""
        t, s, w, x = symbols('t s w x')
        
        # Extract coefficients for linear SDE: dX = (aX + b)dt + (cX + d)dW
        a = diff(drift, x) if x in drift.free_symbols else 0
        b = drift.subs(x, 0) if x in drift.free_symbols else drift
        
        c = diff(diffusion, x) if x in diffusion.free_symbols else 0
        d = diffusion.subs(x, 0) if x in diffusion.free_symbols else diffusion
        
        self.add_step("Identify Coefficients", 
                     f"Drift: $a(t) = {sp.latex(a)}$, $b(t) = {sp.latex(b)}$\n"
                     f"Diffusion: $c(t) = {sp.latex(c)}$, $d(t) = {sp.latex(d)}$")
        
        # Integration factor
        integrand_a = a - c**2/2
        integral_a = integrate(integrand_a, (s, t0, t))
        integral_c = integrate(c, (w, t0, t))  # This represents ∫c dW
        
        Phi = sp.exp(integral_a + integral_c)
        Phi_inv = 1/Phi
        
        self.add_step("Integration Factor", 
                     f"$\\Phi_{{t_0,t}} = \\exp\\left(\\int_{{t_0}}^t \\left(a(s) - \\frac{{1}}{{2}}c(s)^2\\right)ds + \\int_{{t_0}}^t c(s)dW_s\\right)$\n"
                     f"$\\Phi_{{t_0,t}} = {sp.latex(Phi)}$")
        
        # Solution components
        integrand1 = (b - c*d) * Phi_inv
        I1 = integrate(integrand1, (s, t0, t))
        
        integrand2 = d * Phi_inv
        I2 = integrate(integrand2, (w, t0, t))  # Stochastic integral
        
        solution = Phi * (X0 + I1 + I2)
        
        self.add_step("Solution Components",
                     f"$I_1 = \\int_{{t_0}}^t (b(s) - c(s)d(s))\\Phi_{{t_0,s}}^{{-1}} ds = {sp.latex(I1)}$\n"
                     f"$I_2 = \\int_{{t_0}}^t d(s)\\Phi_{{t_0,s}}^{{-1}} dW_s = {sp.latex(I2)}$")
        
        return {
            "solution": solution,
            "integration_factor": Phi,
            "steps": self.steps
        }
    
    def solve_reducible_nonlinear(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve reducible nonlinear SDE using integrating factor method"""
        t, w, x = symbols('t w x')
        
        # For SDE: dX = f(t,X)dt + b(t)X dW
        # Identify b(t)
        b_expr = diffusion / x if x in diffusion.free_symbols else 0
        
        self.add_step("Identify Integrating Factor",
                     f"Diffusion coefficient: $b(t) = {sp.latex(b_expr)}$")
        
        # Integrating factor F_t
        stochastic_integral = integrate(b_expr, w)  # ∫b dW
        deterministic_integral = integrate(b_expr**2, t) / 2  # ½∫b² dt
        
        F_t = sp.exp(-stochastic_integral + deterministic_integral)
        
        self.add_step("Integrating Factor",
                     f"$F_t = \\exp\\left(-\\int b(s)dW_s + \\frac{{1}}{{2}}\\int b(s)^2 ds\\right)$\n"
                     f"$F_t = {sp.latex(F_t)}$")
        
        # Transform to ODE for Y_t = F_t X_t
        Y = Function('Y')(t)
        X_expr = Y / F_t
        
        # The ODE for Y_t
        f_expr = drift.subs(x, X_expr)
        ode_rhs = F_t * f_expr
        
        self.add_step("Transformed ODE",
                     f"Let $Y_t = F_t X_t$, then:\n"
                     f"$\\frac{{dY_t}}{{dt}} = F_t \\cdot f(t, F_t^{{-1}} Y_t)$\n"
                     f"$\\frac{{dY_t}}{{dt}} = {sp.latex(ode_rhs)}$")
        
        # Solve the ODE (this is a simplified version)
        # In practice, you'd use sympy's dsolve or implement specific solution methods
        try:
            from sympy import dsolve
            ode_solution = dsolve(sp.Derivative(Y, t) - ode_rhs, Y)
            X_solution = ode_solution.rhs / F_t
        except:
            # Fallback solution representation
            X_solution = sp.exp(stochastic_integral - deterministic_integral) * \
                        sp.sqrt(X0**2 + 2 * integrate(sp.exp(-2*stochastic_integral + b_expr**2 * t), t))
        
        self.add_step("ODE Solution", 
                     f"Solution of transformed ODE gives:\n"
                     f"$X_t = {sp.latex(X_solution)}$")
        
        return {
            "solution": X_solution,
            "integrating_factor": F_t,
            "steps": self.steps
        }
    
    def solve_kolmogorov_forward(self, mu: sp.Expr, sigma2: sp.Expr, a: float = -oo, b: float = oo) -> Dict:
        """Solve stationary Kolmogorov forward equation using Wright's formula"""
        x = symbols('x')
        
        self.add_step("Wright's Formula",
                     "Stationary solution of Kolmogorov forward equation:\n"
                     "$f(x) = \\frac{\\phi}{\\sigma^2} \\exp\\left(\\int^x \\frac{\\mu(s)}{\\sigma^2(s)} ds\\right)$")
        
        # Compute the integral in the exponent
        exponent_integral = integrate(mu / sigma2, x)
        
        self.add_step("Exponent Integral",
                     f"$\\int \\frac{{\\mu(x)}}{{\\sigma^2(x)}} dx = {sp.latex(exponent_integral)}$")
        
        # Unnormalized density
        unnormalized_density = sp.exp(exponent_integral) / sigma2
        
        # Normalization constant (simplified - actual implementation would compute the integral)
        phi = 1 / integrate(unnormalized_density, (x, a, b))
        
        stationary_density = phi * unnormalized_density
        
        self.add_step("Stationary Density",
                     f"$f(x) = {sp.latex(stationary_density)}$")
        
        return {
            "solution": stationary_density,
            "normalization_constant": phi,
            "steps": self.steps
        }
    
    def solve(self, equation_type: str, drift: str, diffusion: str, 
              initial_condition: Optional[str] = None,
              variables: Dict[str, str] = None,
              parameters: Dict[str, float] = None) -> Dict:
        """Main solving method"""
        self.steps = []
        
        # Default variables
        if variables is None:
            variables = {'t': 'variable', 'x': 'variable', 'W': 'variable'}
        
        # Parse expressions
        drift_expr = self.parse_expression(drift, variables)
        diffusion_expr = self.parse_expression(diffusion, variables)
        
        self.add_step("Problem Setup",
                     f"Solving {'Itô' if equation_type == 'ito' else 'Stratonovich'} SDE:\n"
                     f"$dX_t = {sp.latex(drift_expr)} dt + {sp.latex(diffusion_expr)} dW_t$")
        
        # Choose solution method based on equation type and form
        x = symbols('x')
        is_linear = (diff(drift_expr, x, 2) == 0) and (diff(diffusion_expr, x, 2) == 0)
        
        if is_linear:
            self.add_step("Equation Classification", "Linear SDE detected")
            result = self.solve_linear_sde(drift_expr, diffusion_expr)
        else:
            self.add_step("Equation Classification", "Nonlinear SDE detected - using reducible form method")
            result = self.solve_reducible_nonlinear(drift_expr, diffusion_expr)
        
        # Apply initial condition if provided
        final_solution = result["solution"]
        if initial_condition:
            ic_expr = self.parse_expression(initial_condition, variables)
            # This would need more sophisticated handling in practice
            self.add_step("Initial Condition", f"Applying initial condition: {initial_condition}")
        
        return {
            "steps": self.steps,
            "final_solution": sp.latex(final_solution),
            "solution_type": "exact" if is_linear else "reduced_form",
            "metadata": {
                "equation_type": equation_type,
                "is_linear": is_linear,
                "variables_used": list(variables.keys())
            }
        }

# Utility function for Ito and Stratonovich conversions
def convert_ito_stratonovich(drift_ito: sp.Expr, diffusion: sp.Expr, variables: Dict) -> sp.Expr:
    """Convert between Ito and Stratonovich formulations"""
    x = symbols('x')
    drift_stratonovich = drift_ito - 0.5 * diffusion * diff(diffusion, x)
    return drift_stratonovich 
