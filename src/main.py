import Burgers
import numpy as np

def config_burgers():
    nu = 0.01 / np.pi
    N_u = 2000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    iters = 10000
    config = {'nu': nu, 'N_u': N_u, 'layers': layers, 'iterations': iters, 'learning_rate': 0.001}
    return config

def test_train_split(config, X_star, u_star):
    # create training set
    N_u = config['N_u']
    idx = list(range(0,X_star.shape[0]))
    # print(idx[10])
    np.random.shuffle(idx)
    if N_u < len(idx):
        train_idx = idx[0:N_u]
        test_idx = idx[N_u::]
    else:
        train_idx = idx[0:len(idx)-1]
        test_idx = idx[-1]
    # idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_train = X_star[train_idx, :]
    u_train = u_star[train_idx, :]
    X_test = X_star[test_idx, :]
    u_test = u_star[test_idx, :]
    return X_train, u_train, X_test, u_test

def train_burgers(config, X_train, u_train, X_test, u_test, lb, ub):
    # training
    model = Burgers.BurgersEquation(X_train, u_train, config['layers'], lb, ub, config['learning_rate'])
    model.train(num_iter=config['iterations'])

    u_pred, f_pred = model.predict(X_test)

    error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(u_test, 2)

    lambda_1_value = model.lambda_1.detach().cpu().numpy()
    lambda_2_value = model.lambda_2.detach().cpu().numpy()
    lambda_2_value = np.exp(lambda_2_value)

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - config['nu']) / config['nu'] * 100

    print(f"Error u: {error_u}")
    print(f"Error l1: {error_lambda_1.item()})")
    print(f"Error l2: {error_lambda_2.item()})")

def main():
    config_dict = config_burgers()
    X_star, u_star, lb, ub, X, T = Burgers.load_burgers_data()
    X_train, u_train, X_test, u_test = test_train_split(config_dict, X_star, u_star)
    train_burgers(config_dict, X_train, u_train, X_test, u_test, lb, ub)

if __name__ == "__main__":
    main()