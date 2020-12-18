
import autodiff as ad
x1 = ad.Variable(name = "x1")
x2 = ad.Variable(name = "x2")
y =  x2 + x1

grad, = ad.gradients(y,[x1])
print(y)
executor = ad.Executor([y])
res = executor.run({x1:10,x2:5})
print(res)
