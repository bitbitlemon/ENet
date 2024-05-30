import torch
from torch.optim import Optimizer


class LBFGS_Adam(Optimizer):
    def __init__(self, params, loss_function, gradient_function, lr=0.01, max_iter=100, p1=0.9, p2=0.999):
        defaults = dict(lr=lr, max_iter=max_iter, p1=p1, p2=p2)
        super(LBFGS_Adam, self).__init__(params, defaults)
        self.loss_function = loss_function
        self.gradient_function = gradient_function

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group['lr']
            max_iter = group['max_iter']
            p1 = group['p1']
            p2 = group['p2']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['s'] = []
                    state['y'] = []
                    state['rho'] = []
                    state['ut'] = 0
                    state['wt'] = 0
                    state['m'] = 10  # Number of previous iterations to remember

                state['step'] += 1
                s, y, rho = state['s'], state['y'], state['rho']
                ut, wt = state['ut'], state['wt']

                x = p.data
                g = grad

                q = grad.clone()
                for i in range(len(s) - 1, -1, -1):
                    a = rho[i] * torch.dot(s[i], q)
                    q -= a * y[i]

                r = torch.dot(y[-1], y[-1]) / torch.dot(s[-1], y[-1]) * q

                for i in range(len(s)):
                    beta = rho[i] * torch.dot(y[i], r)
                    r += (alpha[i] - beta) * s[i]

                ut = p2 * ut + (1 - p2) * torch.dot(r, r)
                wt = p1 * wt + (1 - p1) * torch.dot(r, g)
                m_t = ut / (1 - p2 ** state['step'])
                v_t = wt / (1 - p1 ** state['step'])
                step_size = alpha / (torch.sqrt(m_t) + 1e-8)

                x_new = x - step_size * r

                if torch.norm(self.gradient_function(x_new)) < 1e-5:
                    break

                s.append(x_new - x)
                y.append(self.gradient_function(x_new) - g)
                rho.append(1 / torch.dot(s[-1], y[-1]))
                if len(s) > state['m']:
                    s.pop(0)
                    y.pop(0)
                    rho.pop(0)
                x.copy_(x_new)
                g = self.gradient(x)

                state['s'], state['y'], state['rho'] = s, y, rho
                state['ut'], state['wt'] = ut, wt

        return loss

    def gradient(self, x):
        epsilon = 1e-5  # Small perturbation for finite differences
        grad = []
        for i in range(len(x)):
            x_plus = x.clone()
            x_plus[i] += epsilon
            loss_plus = self.loss_function(x_plus)
            x_minus = x.clone()
            x_minus[i] -= epsilon
            loss_minus = self.loss_function(x_minus)
            grad.append((loss_plus - loss_minus) / (2 * epsilon))
        return torch.tensor(grad)
