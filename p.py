import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation
def cost(x):
	m = A.shape[0]
	return 0.5/m *np.linalg.norm(A.dot(x) - b, 2)**2

def grad(x):
	m = A.shape[0]
	return 1/m * A.T.dot(A.dot(x)-b)

def check_grad(x):
	eps = 1e-4 
	g = np.zeros_like(x)
	for i in range (len(x)):

		x1 = x.copy()
		x2 = x.copy()
		x1[i] += eps
		x2[i] -= eps
		g[i] = (cost(x1) - cost(x2))/(2*eps)

	g_grad = grad(x)
	if np.linalg.norm(g-g_grad) > 1e-5:
		print("WARING :CHECK GRADIENT FUNCTION")
def gradient_descent(x_init,learning_rate, iteration):
	x_list = [x_init]
	m = A.shape[0]

	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(x_list[-1])
		if np.linalg.norm(grad(x_new))/m <0.3: #when to stop 
			break
		

		x_list.append(x_new)


	return x_list

A = np.array([[2,7,9,11,16,25,23,22,29,29,35,37,40,46]]).T #diem du lieu co ow truc ox
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).T


fig1 = plt.figure("GD for linear regression")
ax = plt.axes(xlim = (-10,60), ylim=(-1,20)) #dat limit cho x va y =>> truc ox trai tu -10 den 60, truc oy trai tu -1 den 20

plt.plot(A,b,'ro')

#line creat by linaer regression formular
lr = linear_model.LinearRegression()
lr.fit(A,b) #huan luyen, lay du lieu vao
x0_gd = np.linspace(1,46,2)
y0_sklearn = lr.intercept_[0] + lr.coef_[0][0]*x0_gd #trong sklearn thi pt nhap vap la y=b+ax

plt.plot(x0_gd,y0_sklearn, color = "green")


# add one to A
ones = np.ones((A.shape[0],1), dtype =np.int8)
A = np.concatenate((ones,A), axis = 1)

#random initial line
x_init = np.array([[1.],[2.]])

y0_init = x_init[0][0] + x_init[1][0]*x0_gd
plt.plot(x0_gd,y0_init, color = "black")

check_grad(x_init)

#run gradient descent 
iteration = 100
learning_rate = 0.0001
x_list = gradient_descent(x_init, learning_rate, iteration)

#draw x_list (solution by gd
for i in range (len(x_list)):
	y0_x_list = x_list[i][0] + x_list[i][1]*x0_gd #trong sklearn thi pt nhap vap la y=b+ax

	plt.plot(x0_gd,y0_x_list, color = "black", alpha = 0.3)
#draw animation
line, = ax.plot([],[], color = "blue")
def update(i): 
	
	y0_gd = x_list[i][0][0] + x_list[i][1][0]*x0_gd
	line.set_data(x0_gd,y0_gd)
	return line,
iters = np.arange(1,len(x_list), 1)
line_ani = animation.FuncAnimation(fig1, update, iters, interval = 50, blit =True)

#print(len(x_list))
#legen for plot
plt.legend(("value in each GD interation", "Solution by formular", "Inital value for GD"), loc = (0.52,0.01))
ltext = plt.gca().get_legend().get_texts()

plt.title("Gradient descent animation")
plt.show()

# plot cost per iteration to determine when to stop 

#cost_list = []
#iteration_list = []
#for i in range (len(x_list)):
#	iteration_list.append(i)
#	cost_list.append(cost(x_list[i]))

#plt.plot(iteration_list,cost_list)
#plt.xlabel("iteration")
#plt.ylabel("cost value")

#plt.show()
