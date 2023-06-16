from distributions.Lorentz_wrapped_normal import LorentzWrappedNormal
from geoopt import ManifoldParameter
from optim import RiemannianAdam, RiemannianSGD
from os import makedirs
import torch
from os.path import join, exists

def get_origin_kernel_points_lorentz(num_kernel, dim, manifold, max_iter = 100000, verbose = False):
    radius0 = 1.0

    kernel_points = LorentzWrappedNormal(loc = manifold.origin(dim), 
                                         scale = 1, 
                                         manifold = manifold, 
                                         dim=dim).sample([num_kernel - 1])
    kernel_points = ManifoldParameter(kernel_points, manifold = manifold)
    kernel_points.requires_grad_()
    kernel_points.retain_grad()

    fix_point = manifold.origin(dim)
    origin = manifold.origin(dim)

    lr = 1e-3
    alpha = 1
    beta = 10

    optimizer = RiemannianSGD([kernel_points], lr = lr)

    for t in range(max_iter):
        optimizer.zero_grad()
        loss = 0
        for i in range(num_kernel - 1):
            loss += alpha * (1 / manifold.dist(fix_point, kernel_points[i]))
            for j in range(i + 1, num_kernel - 1):
                loss += alpha * (1 / manifold.dist(kernel_points[i], kernel_points[j]))
                # Potential between pairs of kernel points
            loss += beta * manifold.dist(origin, kernel_points[i])
            # Potential between origin and kernel points
        
        loss.backward(retain_graph = True)
        if t % 100 == 0 and verbose:
            print('epoch {}, loss: {}'.format(t, loss.item()))
        optimizer.step()

    return torch.concat([fix_point.view(1, dim), kernel_points.detach()])


def load_kernels(manifold, radius, num_kpoints, dimension, random = False):

    if random:
        kernel_points = manifold.random_normal((num_kpoints, dimension))
        kernel_tangents = manifold.logmap0(kernel_points)
        dis = manifold.dist0(kernel_points).max()
        kernel_tangents *= radius / dis

        return kernel_tangents

    # Kernel directory
    kernel_dir = 'kernels/dispositions'
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # Kernel_file
    kernel_file = join(kernel_dir, 'k_{:03d}_{:d}D.pt'.format(num_kpoints, dimension))

    # Check if already done
    if not exists(kernel_file):
        kernel_points = get_origin_kernel_points_lorentz(num_kernel = num_kpoints, 
                                                         dim = dimension, 
                                                         manifold = manifold, 
                                                         max_iter = 1000, 
                                                         verbose = False)
        kernel_tangents = manifold.logmap0(kernel_points[1:])
        kernel_tangents = torch.concat([torch.zeros(1, dimension), kernel_tangents])

        torch.save(kernel_tangents, kernel_file)

    else:
        kernel_tangents = torch.load(kernel_file)
        kernel_points = manifold.expmap0(kernel_tangents)

    # Scale kernels
    dis = manifold.dist0(kernel_points).max()
    kernel_tangents *= radius / dis

    return kernel_tangents
