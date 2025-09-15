import numpy as np
import pandas as pd
import random
import copy
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pandapower as pp
import pandapower.networks as pn

# =========================================================
# Set seeds
# =========================================================

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
#  Utility Functions
# =========================================================

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

# =========================================================
# Data Processing
# =========================================================

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

# =========================================================
# PyTorch
# =========================================================

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

# =========================================================
# Training
# =========================================================

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

# =========================================================
#  Regular Hyperparameter Tuning (Grid Search / Random Search)
# =========================================================

def hyperparameter_tuning_nn_regular(X_train, Y_train, X_val, Y_val, method='random', n_trials=30):
    """Use regular grid/random search for Neural Network hyperparameter optimization."""
    
    if method == 'grid':
        # Grid search parameters - smaller grid for computational efficiency
        param_grid = {
            'hidden_size_1': [64, 128, 256],
            'hidden_size_2': [32, 64, 128],  
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.0, 0.1, 0.3],
            'batch_size': [32, 64, 128]
        }
        param_combinations = list(ParameterGrid(param_grid))
        print(f"Grid search: Testing {len(param_combinations)} combinations")
        
    else:  # random search
        param_combinations = []
        for _ in range(n_trials):
            params = {
                'hidden_size_1': np.random.choice([32, 64, 128, 256, 512]),
                'hidden_size_2': np.random.choice([32, 64, 128, 256, 512]),
                'learning_rate': np.random.uniform(1e-5, 1e-1),
                'dropout_rate': np.random.uniform(0.0, 0.5),
                'batch_size': np.random.choice([16, 32, 64, 128])
            }
            param_combinations.append(params)
        print(f"Random search: Testing {len(param_combinations)} random combinations")
    
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
    
    best_params = None
    best_fitness = np.inf
    
    pbar = tqdm(param_combinations, desc=f"Regular {method} search")
    for params in pbar:
        fitness = objective_function(params)
        if fitness < best_fitness:
            best_fitness = fitness
            best_params = params
        pbar.set_postfix({'best_fitness': f'{best_fitness:.6f}'})
    
    pbar.close()
    return best_params, best_fitness

def train_svr_model(X_train, Y_train, X_val, Y_val, use_regular_tuning=True):
    """Train SVR model with regular hyperparameter tuning."""
    
    if use_regular_tuning:
        # Use RandomizedSearchCV for SVR
        param_distributions = {
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 'scale'],
            'estimator__epsilon': [0.01, 0.1, 0.2, 0.5, 1.0]
        }
        
        svr_base = MultiOutputRegressor(SVR(kernel='rbf'))
        
        # Combine train and val for cross-validation
        X_tv = np.vstack([X_train, X_val])
        Y_tv = np.vstack([Y_train, Y_val])
        
        print("Running RandomizedSearchCV for SVR...")
        random_search = RandomizedSearchCV(
            svr_base, 
            param_distributions, 
            n_iter=20,  # Number of parameter settings sampled
            cv=3,  # 3-fold cross validation
            scoring='neg_mean_squared_error',
            random_state=SEED,
            n_jobs=-1
        )
        
        random_search.fit(X_tv, Y_tv)
        
        best_svr = random_search.best_estimator_
        best_params = random_search.best_params_
        best_loss = -random_search.best_score_  # Convert back from negative MSE
        
    else:
        best_svr = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale'))
        best_params = {'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1}
        best_loss = np.inf
    
    return best_svr, best_params, best_loss

def train_rf_model(X_train, Y_train, X_val, Y_val, use_regular_tuning=True):
    """Train Random Forest model with regular hyperparameter tuning."""
    
    if use_regular_tuning:
        # Use RandomizedSearchCV for Random Forest
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        }
        
        rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        
        # Combine train and val for cross-validation
        X_tv = np.vstack([X_train, X_val])
        Y_tv = np.vstack([Y_train, Y_val])
        
        print("Running RandomizedSearchCV for Random Forest...")
        random_search = RandomizedSearchCV(
            rf_base, 
            param_distributions, 
            n_iter=20,  # Number of parameter settings sampled
            cv=3,  # 3-fold cross validation
            scoring='neg_mean_squared_error',
            random_state=SEED,
            n_jobs=-1
        )
        
        random_search.fit(X_tv, Y_tv)
        
        best_rf = random_search.best_estimator_
        best_params = random_search.best_params_
        best_loss = -random_search.best_score_  # Convert back from negative MSE
        
    else:
        best_rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        best_params = {'n_estimators': 100, 'max_depth': None}
        best_loss = np.inf
    
    return best_rf, best_params, best_loss

# =========================================================
#  Main Runner
# =========================================================

def regular_opf_runner(cases, n_scenarios, use_regular_tuning=True, nn_search_method='random'):
    """
    Main experimental runner comparing NN, SVR, and RF with regular optimization.
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

        # 1) Neural Network with regular tuning
        print(f"\n--- Training Neural Network with Regular {nn_search_method.title()} Search ---")
        if use_regular_tuning:
            try:
                best_nn_params, best_nn_val_loss = hyperparameter_tuning_nn_regular(
                    X_train, Y_train, X_val, Y_val, method=nn_search_method, n_trials=30
                )
            except Exception as e:
                print(f"Error in NN hyperparameter tuning: {e}")
                best_nn_params = {
                    'hidden_size_1': 128, 'hidden_size_2': 64,
                    'learning_rate': 0.001, 'dropout_rate': 0.1, 'batch_size': 32
                }
                best_nn_val_loss = 1.0
        else:
            best_nn_params = {
                'hidden_size_1': 128, 'hidden_size_2': 64,
                'learning_rate': 0.001, 'dropout_rate': 0.1, 'batch_size': 32
            }
            best_nn_val_loss = 0
        
        # Ensure best_nn_params is valid
        if best_nn_params is None:
            print("Warning: Using fallback parameters for Neural Network")
            best_nn_params = {
                'hidden_size_1': 128, 'hidden_size_2': 64,
                'learning_rate': 0.001, 'dropout_rate': 0.1, 'batch_size': 32
            }
        
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
        print("\n--- Training SVR with Regular Search ---")
        svr_model, svr_params, svr_val_loss = train_svr_model(
            X_train, Y_train, X_val, Y_val, use_regular_tuning=use_regular_tuning
        )
        svr_test_pred = svr_model.predict(X_test)
        svr_test_mse = np.mean((svr_test_pred - Y_test)**2)

        # 3) Random Forest
        print("\n--- Training Random Forest with Regular Search ---")
        rf_model, rf_params, rf_val_loss = train_rf_model(
            X_train, Y_train, X_val, Y_val, use_regular_tuning=use_regular_tuning
        )
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

# =========================================================
# Results
# =========================================================

def visualize_regular_results(results_nn, results_svr, results_rf):
    """Display comparison of NN, SVR, and RF results."""
    methods = ['nn', 'svr', 'rf']
    method_names = ['Neural Network', 'SVR', 'Random Forest']
    results = [results_nn, results_svr, results_rf]
    
    print("\n" + "="*80)
    print("REGULAR-TUNED OPF ML RESULTS COMPARISON")
    print("="*80)
    
    for method, name, result in zip(methods, method_names, results):
        print(f"\n--- {name} Results ---")
        df = pd.DataFrame(result)
        if not df.empty:
            print(df[['Case', 'Test_MSE', 'Val_Loss']].to_string(index=False))
        else:
            print("No results available")
    
    # Summary comparison
    print(f"\n--- SUMMARY: Average Test MSE by Method ---")
    summary_data = []
    for name, result in zip(method_names, results):
        if result:
            avg_mse = np.mean([r['Test_MSE'] for r in result])
            summary_data.append({'Method': name, 'Average_Test_MSE': avg_mse})
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data).sort_values('Average_Test_MSE')
        print(summary_df.to_string(index=False))

# =========================================================
#  Main Execution
# =========================================================

if __name__ == "__main__":
    # Configuration
    cases = ["case30", "case_ieee30", "case39"]  # Power grid test cases
    n_scenarios = 500  # Number of random scenarios per case
    
    print("Starting Regular-OPF Project: NN vs SVR vs RF with Regular Optimization")
    print("="*70)
    
    # Run experiments
    results_nn, results_svr, results_rf = regular_opf_runner(
        cases=cases,
        n_scenarios=n_scenarios,
        use_regular_tuning=True,  # Enable regular hyperparameter optimization
        nn_search_method='random'  # 'random' or 'grid' for neural network
    )
    
    # Display results
    visualize_regular_results(results_nn, results_svr, results_rf)