
import autodiff as ad
x1 = ad.Variable(name = "x1")
x2 = ad.Variable(name = "x2")
y = x1 + x2

grad = ad.gradients(y,[x1,x2])

for grad_dim in grad:
    print(grad_dim)