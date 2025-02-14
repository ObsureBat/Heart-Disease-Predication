"""Enhanced Particle Swarm Optimization (PSO) for hyperparameter tuning"""
import numpy as np
from sklearn.model_selection import cross_val_score
import random
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class Particle:
    def __init__(self, bounds: List[Tuple[float, float]], n_dimensions: int):
        """Initialize particle with position and velocity"""
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) 
                                for i in range(n_dimensions)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(n_dimensions)])
        self.best_position = self.position.copy()
        self.best_score = float('-inf')
        self.current_score = float('-inf')

class AdaptivePSO:
    def __init__(
        self,
        n_particles: int = 30,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1_start: float = 2.5,
        c1_end: float = 0.5,
        c2_start: float = 0.5,
        c2_end: float = 2.5,
        max_iter: int = 50,
        n_iter_no_improve: int = 10,
        tolerance: float = 1e-4
    ):
        """
        Initialize Adaptive PSO optimizer with dynamic parameters
        
        Parameters:
        -----------
        n_particles : int
            Number of particles in the swarm
        w_start, w_end : float
            Initial and final inertia weights
        c1_start, c1_end : float
            Initial and final cognitive parameters
        c2_start, c2_end : float
            Initial and final social parameters
        max_iter : int
            Maximum number of iterations
        n_iter_no_improve : int
            Number of iterations with no improvement before early stopping
        tolerance : float
            Minimum improvement required to reset early stopping counter
        """
        self.n_particles = n_particles
        self.w_start = w_start
        self.w_end = w_end
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end
        self.max_iter = max_iter
        self.n_iter_no_improve = n_iter_no_improve
        self.tolerance = tolerance
        
    def _update_parameters(self, iteration: int) -> Tuple[float, float, float]:
        """Update PSO parameters based on current iteration"""
        progress = iteration / self.max_iter
        
        # Linear decrease of inertia weight
        w = self.w_start - (self.w_start - self.w_end) * progress
        
        # Adaptive cognitive and social parameters
        c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
        c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
        
        return w, c1, c2
    
    def optimize(
        self,
        model_class,
        param_bounds: Dict[str, Tuple[float, float]],
        param_types: Dict[str, str],
        X,
        y,
        cv: int = 3,
        scoring: str = 'roc_auc'
    ) -> Tuple[Dict[str, float], float]:
        """
        Optimize hyperparameters using Adaptive PSO
        
        Parameters:
        -----------
        model_class : class
            Scikit-learn model class
        param_bounds : dict
            Dictionary of parameter bounds (min, max)
        param_types : dict
            Dictionary of parameter types ('int', 'float', 'categorical')
        X : array-like
            Training data
        y : array-like
            Target values
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for cross-validation
        
        Returns:
        --------
        best_params : dict
            Best parameters found
        best_score : float
            Best score achieved
        """
        n_dimensions = len(param_bounds)
        bounds = [(param_bounds[param][0], param_bounds[param][1]) 
                 for param in param_bounds.keys()]
        
        # Initialize particles
        particles = [Particle(bounds, n_dimensions) for _ in range(self.n_particles)]
        global_best_position = None
        global_best_score = float('-inf')
        
        # Early stopping variables
        no_improve_counter = 0
        best_score_history = []
        
        # PSO iterations
        for iteration in range(self.max_iter):
            # Update PSO parameters
            w, c1, c2 = self._update_parameters(iteration)
            
            # Evaluate particles in parallel
            from joblib import Parallel, delayed
            
            def evaluate_particle(particle):
                params = self._position_to_params(
                    particle.position, 
                    list(param_bounds.keys()), 
                    param_types
                )
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = model_class(**params)
                        scores = cross_val_score(
                            model, X, y,
                            cv=cv,
                            scoring=scoring,
                            n_jobs=1  # Use 1 job here since we're parallelizing particles
                        )
                    score = np.mean(scores)
                except Exception:
                    score = float('-inf')
                
                return score
            
            # Parallel evaluation of particles
            scores = Parallel(n_jobs=-1)(
                delayed(evaluate_particle)(particle) 
                for particle in particles
            )
            
            # Update particles with scores
            for particle, score in zip(particles, scores):
                particle.current_score = score
                
                # Update particle's best
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle.position.copy()
            
            # Store best score
            best_score_history.append(global_best_score)
            
            # Check for early stopping
            if len(best_score_history) > 1:
                improvement = best_score_history[-1] - best_score_history[-2]
                if improvement < self.tolerance:
                    no_improve_counter += 1
                else:
                    no_improve_counter = 0
                
                if no_improve_counter >= self.n_iter_no_improve:
                    print(f"Early stopping at iteration {iteration + 1}")
                    break
            
            # Update particles
            for particle in particles:
                r1, r2 = random.random(), random.random()
                
                # Update velocity with adaptive parameters
                particle.velocity = (
                    w * particle.velocity +
                    c1 * r1 * (particle.best_position - particle.position) +
                    c2 * r2 * (global_best_position - particle.position)
                )
                
                # Update position
                particle.position = particle.position + particle.velocity
                
                # Clip position to bounds
                for i in range(n_dimensions):
                    particle.position[i] = np.clip(
                        particle.position[i],
                        bounds[i][0],
                        bounds[i][1]
                    )
        
        # Convert best position to parameters
        best_params = self._position_to_params(
            global_best_position,
            list(param_bounds.keys()),
            param_types
        )
        
        return best_params, global_best_score, best_score_history
    
    def _position_to_params(
        self,
        position: np.ndarray,
        param_names: List[str],
        param_types: Dict[str, str]
    ) -> Dict[str, float]:
        """Convert PSO position to model parameters with proper typing"""
        params = {}
        for i, (name, type_) in enumerate(zip(param_names, param_types.values())):
            if type_ == 'int':
                params[name] = int(round(position[i]))
            elif type_ == 'float':
                params[name] = float(position[i])
            elif type_ == 'categorical':
                params[name] = position[i]
        return params
