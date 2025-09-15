
import numpy as np
import pandas as pd
import random
import copy
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pandapower as pp
import pandapower.networks as pn


# Set seeds
# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# Slime Mold Algorithm

class SlimeMoldAlgorithm:
    """Slime Mold Algorithm for hyperparameter optimization."""
    
    def __init__(self, n_agents=20, max_iter=50, bounds=None, param_types=None):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.bounds = bounds  # [(min, max), (min, max), ...]
        self.param_types = param_types  # ['continuous', 'discrete', 'categorical']
        self.dim = len(bounds)
        self.a = 2  # Controls exploration vs exploitation
        
    def initialize_population(self):
        """Initialize slime mold positions randomly within bounds."""
        population = np.zeros((self.n_agents, self.dim))
        
        for i in range(self.n_agents):
            for j in range(self.dim):
                min_val, max_val = self.bounds[j]
                if self.param_types[j] == 'discrete':
                    population[i, j] = np.random.randint(min_val, max_val + 1)
                else:  # continuous
                    population[i, j] = np.random.uniform(min_val, max_val)
                    
        return population
    
    def decode_params(self, position, param_names, categorical_options=None):
        """Convert numerical position to actual hyperparameters."""
        params = {}
        
        for i, (name, param_type) in enumerate(zip(param_names, self.param_types)):
            if param_type == 'discrete':
                params[name] = int(position[i])
            elif param_type == 'continuous':
                params[name] = position[i]
            elif param_type == 'categorical':
                idx = int(position[i]) % len(categorical_options[name])
                params[name] = categorical_options[name][idx]
                
        return params
    
    def clip_bounds(self, position):
        """Ensure position stays within bounds."""
        for j in range(self.dim):
            min_val, max_val = self.bounds[j]
            position[j] = np.clip(position[j], min_val, max_val)
        return position
    
    def optimize(self, objective_function, param_names, categorical_options=None):
        """Main optimization loop."""
        population = self.initialize_population()
        fitness = np.full(self.n_agents, np.inf)
        
        # Evaluate initial population
        for i in range(self.n_agents):
            params = self.decode_params(population[i], param_names, categorical_options)
            try:
                fitness[i] = objective_function(params)
            except:
                fitness[i] = np.inf
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # SMA main loop
        pbar = tqdm(range(self.max_iter), desc="SMA Optimization")
        
        for iteration in pbar:
            a = 2 - 2 * iteration / self.max_iter  # Decreasing exploration
            
            for i in range(self.n_agents):
                if i < self.n_agents // 2:  # First half: exploitation
                    if np.random.random() < np.tanh(abs(fitness[i] - best_fitness)):
                        # Random position
                        for j in range(self.dim):
                            min_val, max_val = self.bounds[j]
                            population[i, j] = np.random.uniform(min_val, max_val)
                    else:
                        # Move towards best position
                        for j in range(self.dim):
                            population[i, j] = np.random.uniform(-a, a) + best_position[j]
                else:  # Second half: exploration
                    for j in range(self.dim):
                        population[i, j] += np.random.uniform(-1, 1) * a
                
                # Apply bounds
                population[i] = self.clip_bounds(population[i])
                
                # Evaluate new position
                params = self.decode_params(population[i], param_names, categorical_options)
                try:
                    new_fitness = objective_function(params)
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_position = population[i].copy()
                except:
                    pass
            
            pbar.set_postfix({'best_fitness': f'{best_fitness:.6f}'})
        
        pbar.close()
        best_params = self.decode_params(best_position, param_names, categorical_options)
        return best_params, best_fitness


#  Utility Functions

def load_network(case_name):
    """Load a standard network from pandapower."""
    networks = {
        "case30": pn.case30(),
        "case_ieee30": pn.case_ieee30(),
        "case39": pn.case39(),
        "case118": pn.case118(),
        "GBreducednetwork": pn.GBreducednetwork(),
    }
    net = networks.get(case_name, None)
    if net is None:
        raise ValueError(f"Unknown case name: {case_name}")
    return net

def generate_opf_dataset(case_name, n_scenarios=500,
                        p_scale_range=(0.8, 1.2),
                        q_scale_range=(0.8, 1.2)):
    """Generate OPF dataset by randomly scaling loads and solving OPF."""
    base_net = load_network(case_name)
    load_buses = base_net.load.index.values
    gen_buses = base_net.gen.index.values
    
    X_list = []
    Y_list = []
    
    pbar = tqdm(total=n_scenarios, desc=f"Generating scenarios for {case_name}")
    successful_scenarios = 0
    
    for _ in range(n_scenarios):
        net = copy.deepcopy(base_net)
        
        for lb in load_buses:
            rand_p_factor = np.random.uniform(*p_scale_range)
            rand_q_factor = np.random.uniform(*q_scale_range)
            net.load.at[lb, 'p_mw'] *= rand_p_factor
            net.load.at[lb, 'q_mvar'] *= rand_q_factor
        
        try:
            pp.runopp(net, verbose=False)
            successful_scenarios += 1
            
            scenario_load_p = net.load['p_mw'].values
            scenario_load_q = net.load['q_mvar'].values
            X_scenario = np.hstack([scenario_load_p, scenario_load_q])
            
            scenario_gen_p = net.res_gen['p_mw'].values
            scenario_gen_vm = net.res_gen['vm_pu'].values
            Y_scenario = np.hstack([scenario_gen_p, scenario_gen_vm])
            
            X_list.append(X_scenario)
            Y_list.append(Y_scenario)
            
        except pp.optimal_powerflow.OPFNotConverged:
            pass
        
        pbar.update(1)
        pbar.set_postfix({'successful': successful_scenarios})
    
    pbar.close()
    return np.array(X_list), np.array(Y_list)


# Data Processing

def scale_data(inputs, outputs):
    """Apply standard scaling to inputs and outputs."""
    in_scaler = StandardScaler().fit(inputs)
    out_scaler = StandardScaler().fit(outputs)
    inputs_scaled = in_scaler.transform(inputs)
    outputs_scaled = out_scaler.transform(outputs)
    return inputs_scaled, outputs_scaled, in_scaler, out_scaler

def train_val_test_split(X, Y, test_size=0.15, val_size=0.15):
    """Split data into train, validation, and test sets."""
    X_tv, X_test, Y_tv, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=SEED
    )
    
    val_fraction_of_tv = val_size / (1.0 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tv, Y_tv, test_size=val_fraction_of_tv, random_state=SEED
    )
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# PyTorch

class OPFDataset(Dataset):
    """PyTorch Dataset for input-output pairs."""
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class NeuralNetwork(nn.Module):
    """Neural Network for OPF prediction."""
    def __init__(self, x_size, y_size, hidden_sizes, dropout_rate=0.0):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = x_size
        
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_size = h
        
        layers.append(nn.Linear(prev_size, y_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Training


def train_model(model, train_loader, val_loader, epochs, learning_rate, patience=5):
    """Train neural network with early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    use_early_stopping = (val_loader is not None)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if use_early_stopping:
            val_loss = evaluate_model(model, val_loader, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter > patience:
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_loss if use_early_stopping else loss.item()

def evaluate_model(model, data_loader, device):
    """Evaluate model on given data loader."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * len(inputs)
    
    return total_loss / len(data_loader.dataset)

def run_inference_mse(model, inputs, outputs):
    """Compute MSE on entire dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions = model(inputs_tensor)
    
    mse = nn.MSELoss()(predictions, outputs_tensor).item()
    return mse


#  SMA Hyperparameter Tuning


def hyperparameter_tuning_nn_sma(X_train, Y_train, X_val, Y_val, n_agents=15, max_iter=20):
    """Use SMA for Neural Network hyperparameter optimization."""
    
    param_names = ['hidden_size_1', 'hidden_size_2', 'learning_rate', 'dropout_rate', 'batch_size']
    param_types = ['discrete', 'discrete', 'continuous', 'continuous', 'discrete']
    bounds = [
        (32, 512),      # hidden_size_1
        (32, 512),      # hidden_size_2  
        (1e-5, 1e-1),   # learning_rate
        (0.0, 0.5),     # dropout_rate
        (16, 128)       # batch_size
    ]
    
    def objective_function(params):
        try:
            x_size = X_train.shape[1]
            y_size = Y_train.shape[1]
            hidden_sizes = [params['hidden_size_1'], params['hidden_size_2']]
            
            model = NeuralNetwork(x_size, y_size, hidden_sizes, params['dropout_rate'])
            
            train_dataset = OPFDataset(X_train, Y_train)
            val_dataset = OPFDataset(X_val, Y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            val_loss = train_model(
                model, train_loader, val_loader, 
                epochs=30,  # Reduced for faster optimization
                learning_rate=params['learning_rate'], 
                patience=5
            )
            
            return val_loss
            
        except:
            return np.inf
    
    sma = SlimeMoldAlgorithm(n_agents=n_agents, max_iter=max_iter, bounds=bounds, param_types=param_types)
    best_params, best_fitness = sma.optimize(objective_function, param_names)
    
    return best_params, best_fitness

def train_svr_model(X_train, Y_train, X_val, Y_val, use_sma_tuning=True):
    """Train SVR model with optional SMA hyperparameter tuning."""
    
    if use_sma_tuning:
        param_names = ['C', 'gamma', 'epsilon']
        param_types = ['continuous', 'continuous', 'continuous']
        bounds = [
            (0.1, 100),     # C
            (1e-4, 1),      # gamma  
            (0.01, 1.0)     # epsilon
        ]
        
        def svr_objective(params):
            try:
                svr = MultiOutputRegressor(
                    SVR(kernel='rbf', C=params['C'], gamma=params['gamma'], epsilon=params['epsilon'])
                )
                svr.fit(X_train, Y_train)
                Y_val_pred = svr.predict(X_val)
                mse = np.mean((Y_val_pred - Y_val)**2)
                return mse
            except:
                return np.inf
        
        sma = SlimeMoldAlgorithm(n_agents=15, max_iter=25, bounds=bounds, param_types=param_types)
        best_params, best_loss = sma.optimize(svr_objective, param_names)
        
        best_svr = MultiOutputRegressor(
            SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])
        )
    else:
        best_svr = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale'))
        best_params = {'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1}
        best_loss = np.inf
    
    best_svr.fit(X_train, Y_train)
    return best_svr, best_params, best_loss

def train_rf_model(X_train, Y_train, X_val, Y_val, use_sma_tuning=True):
    """Train Random Forest model with optional SMA hyperparameter tuning."""
    
    if use_sma_tuning:
        param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        param_types = ['discrete', 'discrete', 'discrete', 'discrete']
        bounds = [
            (50, 300),      # n_estimators
            (5, 30),        # max_depth
            (2, 20),        # min_samples_split
            (1, 10)         # min_samples_leaf
        ]
        
        def rf_objective(params):
            try:
                rf = RandomForestRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    random_state=SEED,
                    n_jobs=-1
                )
                rf.fit(X_train, Y_train)
                Y_val_pred = rf.predict(X_val)
                mse = np.mean((Y_val_pred - Y_val)**2)
                return mse
            except:
                return np.inf
        
        sma = SlimeMoldAlgorithm(n_agents=15, max_iter=25, bounds=bounds, param_types=param_types)
        best_params, best_loss = sma.optimize(rf_objective, param_names)
        
        best_rf = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=SEED,
            n_jobs=-1
        )
    else:
        best_rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        best_params = {'n_estimators': 100, 'max_depth': None}
        best_loss = np.inf
    
    best_rf.fit(X_train, Y_train)
    return best_rf, best_params, best_loss


#  Main Runner

def sma_opf_runner(cases, n_scenarios, use_sma_tuning=True):
    """
    Main experimental runner comparing NN, SVR, and RF with SMA optimization.
    """
    results_nn = []
    results_svr = []
    results_rf = []

    case_pbar = tqdm(cases, desc="Processing cases")
    
    for case in case_pbar:
        case_pbar.set_description(f"Processing {case}")
        print(f"\n>>>> Processing {case} with up to {n_scenarios} scenarios...\n")
        
        X_raw, Y_raw = generate_opf_dataset(case, n_scenarios=n_scenarios)

        if len(X_raw) < 30:
            print(f"Not enough data for {case} (got {len(X_raw)}). Skipping.")
            continue

        # Scale data
        X_scaled, Y_scaled, in_scaler, out_scaler = scale_data(X_raw, Y_raw)

        # Train/Val/Test split
        X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(
            X_scaled, Y_scaled, test_size=0.15, val_size=0.15
        )
        print(f"Data Split => Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # 1) Neural Network with SMA tuning
        print("\n--- Training Neural Network with SMA ---")
        if use_sma_tuning:
            best_nn_params, best_nn_val_loss = hyperparameter_tuning_nn_sma(
                X_train, Y_train, X_val, Y_val, n_agents=15, max_iter=20
            )
        else:
            best_nn_params = {
                'hidden_size_1': 128, 'hidden_size_2': 64,
                'learning_rate': 0.001, 'dropout_rate': 0.1, 'batch_size': 32
            }
            best_nn_val_loss = 0
        
        # Train final NN on train+val
        X_tv = np.vstack([X_train, X_val])
        Y_tv = np.vstack([Y_train, Y_val])
        
        final_nn = NeuralNetwork(
            X_tv.shape[1], Y_tv.shape[1], 
            [best_nn_params['hidden_size_1'], best_nn_params['hidden_size_2']], 
            best_nn_params['dropout_rate']
        )
        
        ds_tv = OPFDataset(X_tv, Y_tv)
        loader_tv = DataLoader(ds_tv, batch_size=best_nn_params['batch_size'], shuffle=True)
        
        _ = train_model(final_nn, loader_tv, None, epochs=50, 
                       learning_rate=best_nn_params['learning_rate'], patience=5)
        
        nn_test_mse = run_inference_mse(final_nn, X_test, Y_test)

        # 2) Support Vector Regression
        print("\n--- Training SVR with SMA ---")
        svr_model, svr_params, svr_val_loss = train_svr_model(
            X_train, Y_train, X_val, Y_val, use_sma_tuning=use_sma_tuning
        )
        svr_model.fit(X_tv, Y_tv)  # Retrain on full data
        svr_test_pred = svr_model.predict(X_test)
        svr_test_mse = np.mean((svr_test_pred - Y_test)**2)

        # 3) Random Forest
        print("\n--- Training Random Forest with SMA ---")
        rf_model, rf_params, rf_val_loss = train_rf_model(
            X_train, Y_train, X_val, Y_val, use_sma_tuning=use_sma_tuning
        )
        rf_model.fit(X_tv, Y_tv)  # Retrain on full data
        rf_test_pred = rf_model.predict(X_test)
        rf_test_mse = np.mean((rf_test_pred - Y_test)**2)

        # Store results
        results_nn.append({
            'Case': case, 'Test_MSE': nn_test_mse, 'Best_Params': best_nn_params, 'Val_Loss': best_nn_val_loss
        })
        results_svr.append({
            'Case': case, 'Test_MSE': svr_test_mse, 'Best_Params': svr_params, 'Val_Loss': svr_val_loss
        })
        results_rf.append({
            'Case': case, 'Test_MSE': rf_test_mse, 'Best_Params': rf_params, 'Val_Loss': rf_val_loss
        })

    case_pbar.close()
    return results_nn, results_svr, results_rf


# Results


def visualize_sma_results(results_nn, results_svr, results_rf):
    """Display comparison of NN, SVR, and RF results."""
    methods = ['nn', 'svr', 'rf']
    method_names = ['Neural Network', 'SVR', 'Random Forest']
    results = [results_nn, results_svr, results_rf]
    
    
    print("SMA-OPTIMIZED OPF")
    
    
    for method, name, result in zip(methods, method_names, results):
        print(f"\n--- {name} Results ---")
        df = pd.DataFrame(result)
        if not df.empty:
            print(df[['Case', 'Test_MSE', 'Val_Loss']].to_string(index=False))
        else:
            print("No results available")
    
    # Summary comparison
    print(f"\nSummary")
    summary_data = []
    for name, result in zip(method_names, results):
        if result:
            avg_mse = np.mean([r['Test_MSE'] for r in result])
            summary_data.append({'Method': name, 'Average_Test_MSE': avg_mse})
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data).sort_values('Average_Test_MSE')
        print(summary_df.to_string(index=False))


#  Main Execution

if __name__ == "__main__":
    # Configuration
    cases = ["case30", "case_ieee30", "case39", "case118"]  # Power grid test cases
    n_scenarios = 500  # Number of random scenarios per case
    
    print("SMA-OPF Project")
    
    
    # Run experiments
    results_nn, results_svr, results_rf = sma_opf_runner(
        cases=cases,
        n_scenarios=n_scenarios,
        use_sma_tuning=True  # Enable SMA hyperparameter optimization
    )
    
    # Display results
    visualize_sma_results(results_nn, results_svr, results_rf)

    
    