# Defining the Durgers Eqaution, Losses, and Data input to fit a neural field to the approximate solutions

import numpy as np
import torch
import network
import scipy.io
from scipy.interpolate import griddata


class BurgersEquation(network.PhysicsInformedNN):
    def __init__(self, X, u, layers, lb, ub, learning_rate=0.001):
        super(BurgersEquation, self).__init__(X, u, layers, device='cpu')
        self.learning_rate = learning_rate
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(self.device)
        self.ub = torch.tensor(ub).float().to(self.device)

        # settings
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(self.device)
        self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(self.device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        # deep neural networks

        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # # optimizers: using the same settings
        # self.optimizer = torch.optim.LBFGS(
        #     self.dnn.parameters(),
        #     lr=1.0,
        #     max_iter=50000,
        #     max_eval=50000,
        #     history_size=50,
        #     tolerance_grad=1e-5,
        #     tolerance_change=1.0 * np.finfo(float).eps,
        #     line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        # )

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=self.learning_rate)
        # self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def loss_func(self):
        u_pred_loss = self.net_u(self.x, self.t)
        f_pred_loss = self.net_f(self.x, self.t)
        loss = torch.mean((self.u - u_pred_loss) ** 2) + torch.mean(f_pred_loss ** 2)
        # self.optimizer.zero_grad()
        # loss.backward()

        # self.iter += 1
        # if self.iter % 100 == 0:
        #     print(
        #         'Loss: %e, l1: %.5f, l2: %.5f' %
        #         (
        #             loss.item(),
        #             self.lambda_1.item(),
        #             torch.exp(self.lambda_2.detach()).item()
        #         )
        #     )
        return loss

    def train(self, num_iter):
        self.dnn.train()
        for epoch in range(num_iter):
            loss = self.loss_func()

            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()

            # convert to f string.
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Lambda_1: {self.lambda_1.item():.4f}, "
                  f"Lambda_2: {torch.exp(self.lambda_2).item():.4f}")

        # Backward and optimize
        self.optimizer_Adam.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

def load_burgers_data(path_to_data='../data/burgers_shock.mat'):
    data = scipy.io.loadmat(path_to_data)

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X_grid, T_grid = np.meshgrid(x, t)

    X_star = np.hstack((X_grid.flatten()[:, None], T_grid.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # boundary conditions
    lb = X_star.min(0)
    ub = X_star.max(0)
    return X_star, u_star, lb, ub, X_grid, T_grid

if __name__ == "__main__":
    nu = 0.01 / np.pi
    N_u = 2000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    # create training set
    X_star, u_star, lb, ub, X, T = load_burgers_data()
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]

    # training
    model = BurgersEquation(X_u_train, u_train, layers, lb, ub)
    model.train(200)

    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    lambda_1_value = model.lambda_1.detach().cpu().numpy()
    lambda_2_value = model.lambda_2.detach().cpu().numpy()
    lambda_2_value = np.exp(lambda_2_value)

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

    print(f"Error u: {error_u}")
    print(f"Error l1: {error_lambda_1.item()})")
    print(f"Error l2: {error_lambda_2.item()})")