import sympy as sp
from sympy import symbols, Function, exp, integrate, diff, sqrt, pi, oo, Eq, solve
from typing import Dict, List, Any, Optional
import re

class SDESolver:
    """
    Symbolic solver for Stochastic Differential Equations (SDEs)
    Based on the work of Chu-Ching Huang (2011)
    Extended with additional solution methods
    """
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, title: str, content: str):
        """Add a step to the solution process"""
        cleaned_content = self.clean_latex_content(content)
        self.steps.append({"title": title, "content": cleaned_content})
    
    def clean_latex_content(self, content: str) -> str:
        """Clean excessive $ in LaTeX content"""
        if not content:
            return content
            
        # Nettoyage des séquences de $ ou \[\]
        cleaned = re.sub(r'\$\s*\\\s*\$', '', content)
        cleaned = re.sub(r'\$\s*\$', '', cleaned)
        cleaned = re.sub(r'\\\[\s*\\\]', '', cleaned)
        
        # Supprime les délimiteurs de début/fin si l'expression est entre $...$ ou \[...\]
        cleaned = re.sub(r'^\$|\$$', '', cleaned)
        cleaned = re.sub(r'^\\\[|\\\]$', '', cleaned)
        return cleaned.strip()
    
    def parse_expression(self, expr_str: str, variables: Dict[str, str]) -> sp.Expr:
        """Parse string expression to sympy expression"""
        sym_map = {}
        for var_name, var_type in variables.items():
            if var_type == "variable":
                sym_map[var_name] = symbols(var_name)
            elif var_type == "function":
                sym_map[var_name] = Function(var_name)(symbols('t'))
        
        common_symbols = {
            't': symbols('t'), 'x': symbols('x'), 'y': symbols('y'), 'z': symbols('z'),
            'w': symbols('w'), 'W': symbols('W'), 'dt': symbols('dt'), 'dw': symbols('dw'),
            'dW': symbols('dW'), 'alpha': symbols('alpha'), 'beta': symbols('beta'),
            'sigma': symbols('sigma'), 'mu': symbols('mu'), 'theta': symbols('theta'),
            'a': symbols('a'), 'b': symbols('b'), 'c': symbols('c'), 'd': symbols('d'),
            'sqrt': sqrt, 'exp': exp, 'integrate': integrate, 'diff': diff
        }
        sym_map.update(common_symbols)
        
        try:
            return sp.sympify(expr_str, locals=sym_map)
        except:
            return self._build_expression_manual(expr_str, sym_map)
    
    def _build_expression_manual(self, expr_str: str, sym_map: Dict) -> sp.Expr:
        """Manual expression building for complex cases"""
        try:
            # Sécurité: utiliser eval pour les fonctions mathématiques communes
            allowed_globals = {"__builtins__": None, "sqrt": sqrt, "exp": exp}
            # Également permettre d'accéder aux symboles définis dans sym_map
            return eval(expr_str, allowed_globals, sym_map)
        except Exception:
            # Fallback to simple sympify if manual eval fails
            return sp.sympify(expr_str)
    
    def solve_separable_sde(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve SDE using separation of variables method"""
        t, x, w = symbols('t x w')
        
        self.add_step("Separation of Variables Method", "For SDE: $dX_t = \\mu(t,X)dt + \\sigma(t,X)dW_t$")
        
        try:
            # Cas spécifique pour l'équation linéaire (intégration par facteur intégrant)
            # Cette détection est une heuristique et pourrait être améliorée.
            if x in drift.free_symbols and x in diffusion.free_symbols:
                # Vérification si l'équation est linéaire du type dX_t = (a(t)X + b(t))dt + (c(t)X + d(t))dW_t
                if diff(drift, x, 2) == 0 and diff(diffusion, x, 2) == 0:
                    self.add_step("Case Detection", "Equation is linear, deferring to linear solver approach.")
                    return self.solve_linear_sde(drift, diffusion, t0, X0)

            # Essayer de séparer les variables (pas toujours possible symboliquement)
            g_x = 1 / diffusion.subs(t, 1) if not diffusion.free_symbols.difference({x}) else 0
            f_t = drift.subs(x, 1) if not drift.free_symbols.difference({t}) else 0
            
            if g_x != 0 and f_t != 0:
                self.add_step("Separation Attempt", "Attempting separation of variables: $\\frac{dX_t}{\\sigma(X)} = \\frac{\\mu(X)}{\\sigma(X)}dt + dW_t$")
                
                # C'est souvent plus complexe qu'une simple séparation pour les SDEs.
                # Nous nous reposons sur la méthode de Huang pour la classe "linéaire" ou "réductible".
                
                # Fallback to the reducible method as a general non-linear attempt
                return self.solve_reducible_nonlinear(drift, diffusion, t0, X0)

            # Si l'équation n'est pas séparable trivialement et non linéaire.
            return self.solve_reducible_nonlinear(drift, diffusion, t0, X0)

        except Exception as e:
            self.add_step("Separation Method Failed", f"Could not apply separation method: {str(e)}")
            return self.solve_reducible_nonlinear(drift, diffusion, t0, X0)
    
    def solve_linear_sde(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve linear SDE using the method from Huang's paper (Itô's formula applied to X*Phi_inv)"""
        t, s, w, x = symbols('t s w x')
        
        # Identification des coefficients de la SDE Linéaire: dX_t = (a(t)X + b(t))dt + (c(t)X + d(t))dW_t
        try:
            # a(t) = d/dx (drift) | x=...
            a = diff(drift, x).subs(x, x) if x in drift.free_symbols else sp.sympify(0)
            # b(t) = drift | x=0
            b = drift.subs(x, 0) if x in drift.free_symbols else drift
            # c(t) = d/dx (diffusion) | x=...
            c = diff(diffusion, x).subs(x, x) if x in diffusion.free_symbols else sp.sympify(0)
            # d(t) = diffusion | x=0
            d = diffusion.subs(x, 0) if x in diffusion.free_symbols else diffusion
        except:
            # Cas où les expressions ne dépendent pas de x
            a, c = sp.sympify(0), sp.sympify(0)
            b, d = drift, diffusion

        step1_content = f"SDE Linéaire: $dX_t = (a(t)X_t + b(t))dt + (c(t)X_t + d(t))dW_t$\n$a(t) = {sp.latex(a)}$, $b(t) = {sp.latex(b)}$\n$c(t) = {sp.latex(c)}$, $d(t) = {sp.latex(d)}$"
        self.add_step("Identify Coefficients", step1_content)
        
        # Facteur intégrant Phi
        integrand_a = a - c**2/2
        integral_a = integrate(integrand_a, (s, t0, t))
        integral_c = integrate(c, (s, t0, t)) # Corrigé: L'intégrale stochastique est par rapport à dW_s, mais sympy gère l'intégration indéfinie
        
        # L'utilisation de 'w' comme variable d'intégration pour dW_s est symbolique ici
        integral_c_dW = integrate(c, s).subs(s, t) * symbols('W_t') # Simplification symbolique
        
        Phi = sp.exp(integral_a + integral_c_dW)
        Phi_inv = 1/Phi
        
        self.add_step("Integration Factor", f"$\\Phi_{{t_0,t}} = \\exp\\left(\\int_{{t_0}}^t \\left(a(s) - \\frac{{1}}{{2}}c(s)^2\\right)ds + \\int_{{t_0}}^t c(s)dW_s\\right)$")
        
        # Calcul des intégrales stochastiques et déterministes (symboliquement difficile)
        # On va laisser les intégrales dans la solution pour la clarté.
        
        solution_formula = f"X_t = \Phi_{{t_0,t}} \\left( X_0 + \\int_{{t_0}}^t (b(s) - c(s)d(s))\\Phi_{{t_0,s}}^{{-1}} ds + \\int_{{t_0}}^t d(s)\\Phi_{{t_0,s}}^{{-1}} dW_s \\right)"
        
        # Construction de la solution symbolique (difficile avec les intégrales stochastiques)
        # Sympy ne peut pas calculer ces intégrales stochastiques. On retourne la formule générale simplifiée.
        
        # Représentation de l'intégrale stochastique
        I_stoch_sym = Function('\\int_{t_0}^t d(s)\\Phi_{t_0,s}^{{-1}} dW_s')(t)
        I_det_sym = Function('\\int_{t_0}^t (b(s) - c(s)d(s))\\Phi_{t_0,s}^{{-1}} ds')(t)
        
        solution_symbolic = Phi * (X0 + I_det_sym + I_stoch_sym)
        
        self.add_step("Solution (General Formula)", f"$X_t = {sp.latex(solution_symbolic)}$")
        
        return {
            "solution": solution_symbolic,
            "solution_type": "linear_general_formula",
            "integration_factor": Phi,
            "steps": self.steps
        }
    
    def solve_kolmogorov_forward(self, drift: sp.Expr, diffusion: sp.Expr) -> Dict:
        """Solve stationary Kolmogorov forward equation using Wright's formula"""
        t, x = symbols('t x')
        
        step1_content = "Stationary solution of Kolmogorov forward equation (Fokker-Planck):\n$\\frac{\\partial p}{\\partial t} = -\\frac{\\partial}{\\partial x}[\\mu(x)p] + \\frac{1}{2}\\frac{\\partial^2}{\\partial x^2}[\\sigma^2(x)p]$"
        self.add_step("Kolmogorov Forward Equation (Fokker-Planck)", step1_content)
        
        step2_content = "For stationary solution $p_{st}(x)$ ($\\frac{\\partial p}{\\partial t} = 0$), the formula is:\n$p_{st}(x) = \\frac{C}{\\sigma^2(x)} \\exp\\left(2\\int^x \\frac{\\mu(s)}{\\sigma^2(s)} ds\\right)$"
        self.add_step("Stationary Solution Formula", step2_content)
        
        try:
            # Calcul de l'intégrale
            integrand_exp = 2 * drift / (diffusion**2)
            exponent_integral = integrate(integrand_exp, x)
            self.add_step("Exponent Integral", f"$2\\int \\frac{{\\mu(x)}}{{\\sigma^2(x)}} dx = {sp.latex(exponent_integral)}$")
            
            # Calcul de la densité non normalisée
            unnormalized_density = sp.exp(exponent_integral) / (diffusion**2)
            C = symbols('C')
            stationary_density = C * unnormalized_density
            
            step4_content = f"Unnormalized Stationary Density:\n$p_{{st}}(x) = {sp.latex(stationary_density)}$"
            self.add_step("Unnormalized Density Result", step4_content)
            
            return {
                "solution": stationary_density,
                "normalization_constant": C,
                "solution_type": "stationary_distribution",
                "steps": self.steps
            }
        except Exception as e:
            self.add_step("Integration Failed", f"Could not compute the integral: {str(e)}")
            C = symbols('C')
            # Retourne la formule non intégrée pour l'affichage
            general_solution = C * sp.exp(2 * integrate(drift / (diffusion**2), x)) / (diffusion**2)
            
            return {
                "solution": general_solution,
                "normalization_constant": C,
                "solution_type": "stationary_distribution_general",
                "steps": self.steps
            }
    
    def solve_reducible_nonlinear(self, drift: sp.Expr, diffusion: sp.Expr, t0: float = 0, X0: float = 0) -> Dict:
        """Solve SDE reducible to a deterministic ODE via a change of variables (Ex: dX_t = f(X)dW_t)"""
        t, w, x = symbols('t w x')
        
        # Cette méthode est souvent appliquée aux SDEs dX_t = f(t, X)dt + g(t)X dW_t
        # ou aux SDEs qui se transforment bien (ex: exponentielles)
        
        try:
            # Hypothèse: dX_t = f(t, X)dt + g(t)X dW_t (Réductible au log)
            # Ici, nous allons simplifier en supposant un cas où la substitution Y=log(X) fonctionne
            
            # Si le terme de diffusion est de la forme sigma*X (Mouvement Brownien Géométrique)
            if diffusion.free_symbols.difference({x, t}) == set() and diff(diffusion, x, 2) == 0:
                # Calcul de g(t) = c(t) = diffusion/X
                g_t = diff(diffusion, x).subs(x, x)
            else:
                self.add_step("Reducible Method Failed", "Diffusion term is not of the form g(t)X. Fallback to general form.")
                raise ValueError("Diffusion term too complex for this reduction method.")
            
            # Formule de transformation pour Y = log(X)
            # dY_t = ( (f(t,X)/X) - 0.5*g(t)^2 ) dt + g(t) dW_t
            
            drift_over_x = drift / x
            drift_new = drift_over_x - 0.5 * g_t**2
            
            integral_drift = integrate(drift_new, t)
            integral_diffusion = integrate(g_t, w)
            
            # La solution pour Y_t = log(X_t)
            Y_solution_sym = symbols('C1') + integral_drift + integral_diffusion
            
            # Solution pour X_t
            X_solution = exp(Y_solution_sym)
            
            self.add_step("Transformation to Logarithmic Process", 
                          f"Using Itô's formula for $Y_t = \ln(X_t)$.\n$dY_t = \\left( \\frac{{\\mu(t,X)}}{{X}} - \\frac{{1}}{{2}}\\sigma(t,X)^2 \\right) dt + \\frac{{\\sigma(t,X)}}{{X}} dW_t$")
            
            self.add_step("Integration Result", f"$Y_t = {sp.latex(Y_solution_sym)}$")
            
            return {
                "solution": X_solution.subs(symbols('C1'), sp.log(X0)), # Remplace C1 par log(X0)
                "solution_type": "reducible_logarithmic",
                "steps": self.steps
            }
        
        except Exception as e:
            # Fallback général pour les cas non résolus
            self.add_step("Reducible Method Failed", f"Could not apply reduction to log process: {str(e)}")
            
            # Tente la solution standard pour dX_t = f(X)dW_t -> 1/f(X)dX_t = dW_t
            # (souvent incorrecte à cause du terme Ito)
            
            X_solution_approx = X0 * sp.exp(integrate(drift, t) - integrate(diffusion**2, t)/2 + integrate(diffusion, w))
            
            return {
                "solution": X_solution_approx,
                "solution_type": "reducible_failed_approx",
                "steps": self.steps
            }

    
    def solve(self, equation_type: str, drift: str, diffusion: str, 
              initial_condition: Optional[str] = None,
              variables: Dict[str, str] = None,
              parameters: Dict[str, float] = None) -> Dict:
        """Main solving method with enhanced capabilities"""
        self.steps = []
        
        if variables is None:
            variables = {'t': 'variable', 'x': 'variable', 'W': 'variable'}
        
        # 1. Parse les expressions
        drift_expr = self.parse_expression(drift, variables)
        diffusion_expr = self.parse_expression(diffusion, variables)
        
        step_content = f"Solving {'Itô' if equation_type == 'ito' else 'Stratonovich'} SDE:\n$dX_t = {sp.latex(drift_expr)} dt + {sp.latex(diffusion_expr)} dW_t$"
        self.add_step("Problem Setup", step_content)
        
        x, t = symbols('x t')
        is_kolmogorov = ("kolmogorov" in drift.lower() or "stationary" in drift.lower() or
                        (not x in drift_expr.free_symbols and not t in drift_expr.free_symbols and not x in diffusion_expr.free_symbols and not t in diffusion_expr.free_symbols))
        
        # 2. Classification et résolution
        if is_kolmogorov or not initial_condition: # Si on cherche une distribution stationnaire ou si X0 est manquant
            self.add_step("Equation Classification", "Kolmogorov Forward Equation (Stationary Density) approach detected or no initial condition given.")
            result = self.solve_kolmogorov_forward(drift_expr, diffusion_expr)
            solution_type = "stationary_distribution"
        else:
            # Convertir Stratonovich en Itô si nécessaire (simplification: on suppose la formule Itô)
            if equation_type == 'stratonovich':
                self.add_step("Itô-Stratonovich Conversion", "Converting Stratonovich to Itô: $\\mu_{Ito} = \\mu_{Strat} + \\frac{1}{2}\\sigma(x)\\sigma'(x)$")
                drift_expr = drift_expr + 0.5 * diffusion_expr * diff(diffusion_expr, x)
                self.add_step("Itô Form", f"$dX_t = {sp.latex(drift_expr)} dt + {sp.latex(diffusion_expr)} dW_t$")

            # Test de linéarité
            is_linear = (diff(drift_expr, x, 2) == 0) and (diff(diffusion_expr, x, 2) == 0)
            
            if is_linear:
                self.add_step("Equation Classification", "Linear SDE detected")
                result = self.solve_linear_sde(drift_expr, diffusion_expr, float(initial_condition.split('=')[0].strip()), float(initial_condition.split('=')[1].strip()))
                solution_type = "linear_general_formula"
            else:
                self.add_step("Equation Classification", "Nonlinear SDE detected - trying various methods")
                # On utilise solve_separable_sde qui contient des heuristiques
                result = self.solve_separable_sde(drift_expr, diffusion_expr, float(initial_condition.split('=')[0].strip()), float(initial_condition.split('=')[1].strip()))
                solution_type = result.get("solution_type", "exact_approx")
        
        final_solution = result["solution"]
        
        # 3. Application de la Condition Initiale (pour les SDEs)
        if initial_condition and not solution_type.startswith("stationary"):
            try:
                # Extraction X0 et t0 de la condition initiale (ex: X(0)=100)
                ic_match = re.search(r'X\s*\(\s*(\d*\.?\d*)\s*\)\s*=\s*(\d*\.?\d*)', initial_condition)
                if ic_match:
                    t0_val = float(ic_match.group(1))
                    x0_val = float(ic_match.group(2))
                else:
                    t0_val = parameters.get('t0', 0) if parameters else 0
                    x0_val = parameters.get('x0', 1) if parameters else 1
                
                # Symbole constant d'intégration (si présent)
                C_sym = symbols('C1') # Utilisé dans la fonction de log
                
                if C_sym in final_solution.free_symbols:
                    # Pour les cas où la constante C1 est encore là
                    ic_eq = Eq(final_solution.subs(t, t0_val), x0_val)
                    C_solution = solve(ic_eq, C_sym)
                    if C_solution:
                        final_solution = final_solution.subs(C_sym, C_solution[0])
                        self.add_step("Apply Initial Condition", f"Using $X({t0_val}) = {x0_val}$ to determine constant $C_1$")
                    else:
                         self.add_step("Initial Condition Application", "Could not solve for integration constant C1.")

            except Exception as e:
                self.add_step("Initial Condition Application", f"Could not apply initial condition: {str(e)}")
        
        final_solution_latex = self.clean_latex_content(sp.latex(final_solution))
        
        return {
            "steps": self.steps,
            "final_solution": final_solution_latex,
            "solution_type": solution_type,
            "metadata": {
                "equation_type": equation_type,
                "is_kolmogorov": solution_type.startswith("stationary"),
                "variables_used": list(variables.keys())
            }
        }

def convert_ito_stratonovich(drift_ito: sp.Expr, diffusion: sp.Expr, variables: Dict) -> sp.Expr:
    """Convert between Ito and Stratonovich formulations (only drift)"""
    x = symbols('x')
    drift_stratonovich = drift_ito - 0.5 * diffusion * diff(diffusion, x)
    return drift_stratonovich
