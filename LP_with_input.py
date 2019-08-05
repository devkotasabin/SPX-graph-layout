#import sys
#sys.path.append('/cm/shared/uaapps/cplex/12.6.2/cplex/python/2.6/x86-64_linux/')
import cplex
from cplex.exceptions import CplexError
import sys

# #Input 1
# ax = -1
# ay = 1

# bx = 1
# by = -1

# cx = 1
# cy = 1

# dx = -1
# dy = -1

# #Input 2
# #Long version of input 1
# ax = -10
# ay = 10

# bx = 10
# by = -10

# cx = 10
# cy = 10

# dx = -10
# dy = -10

# #Input 3
# # Rotated both edges of input 1 by 22.5 deg
# ax = -1.3066
# ay = 0.5412

# bx = 1.3066
# by = -0.5412

# cx = 1.3066
# cy = 0.5412

# dx = -1.3066
# dy = -0.5412

# #Input 4
# # Rotated both edges of input 2 by 22.5 deg
# ax = -13.0656
# ay = 5.4120

# bx = 13.0656
# by = -5.4120

# cx = 13.0656
# cy = 5.4120

# dx = -13.0656
# dy = -5.4120

# #Input 5
# #Axis-oriented lines
# ax = 0
# ay = 3

# bx = 0
# by = -3

# cx = -3
# cy = 0

# dx = 3
# dy = 0

# #Input 6
# #Long version of input 5
# ax = 0
# ay = 30

# bx = 0
# by = -30

# cx = -30
# cy = 0

# dx = 30
# dy = 0

# #Input 7
# #T shape balanced
# ax = 0
# ay = 3

# bx = 0
# by = -3

# cx = -3
# cy = 2

# dx = 3
# dy = 2

# #Input 7_2
# #T shape imbalanced
# ax = 0
# ay = 3

# bx = 0
# by = -3

# cx = -2
# cy = 2

# dx = 4
# dy = 2

# #Input 8
# #T shape balanced long
# ax = 0
# ay = 30

# bx = 0
# by = -30

# cx = -30
# cy = 20

# dx = 30
# dy = 20


# #Input 9
# #T shape imbalanced long
# ax = 0
# ay = 30

# bx = 0
# by = -30

# cx = -20
# cy = 20

# dx = 40
# dy = 20

# #Input 10
# #T shape imbalanced another
# ax = 0
# ay = 30

# bx = 0
# by = -30

# cx = -10
# cy = 20

# dx = 50
# dy = 20

# #Input 11
# #T shape imbalanced another small
# ax = 0
# ay = 3

# bx = 0
# by = -3

# cx = -1
# cy = 2

# dx = 5
# dy = 2

# #Input 12
# #Imbalanced intersection
# ax = 2
# ay = -2

# bx = -8
# by = 8

# cx = -2
# cy = 0

# dx = 10
# dy = 0

# #Input 13
# #Imbalanced intersection long
# ax = 20
# ay = -20

# bx = -80
# by = 80

# cx = -20
# cy = 0

# dx = 100
# dy = 0

# #Input 14
# #Imbalanced intersection smaller angle
# ax = 2.6131
# ay = -1.0824

# bx = -10.4525
# by = 4.3296

# cx = -2
# cy = 0

# dx = 10
# dy = 0

# #Input 15
# #Long version of input 14
# ax = 26.1313
# ay = -10.8239

# bx = -104.5250
# by = 43.2957

# cx = -20
# cy = 0

# dx = 100
# dy = 0

# #Input 16
# #T-shape imbalanced with different edge lengths
# ax = 0
# ay = 3

# bx = 0
# by = -3

# cx = -2
# cy = 2

# dx = 1
# dy = 2


# #Input 17
# #Long version of Input 16
# ax = 0
# ay = 30

# bx = 0
# by = -30

# cx = -20
# cy = 20

# dx = 10
# dy = 20

# #Input 18
# #Rotated version of Input 16
# ax = 0
# ay = 3

# bx = 0
# by = -3

# cx = -1.4142
# cy = 3.4142

# dx = 0.7071
# dy = 1.2929

# #Input 19
# #T-shape with unequal edge length
ax = 0
ay = 30

bx = 0
by = -30

cx = -2
cy = 25

dx = 5
dy = 25

# #Input 20
# # No intersection
# ax = 0
# ay = 0

# bx = 1
# by = 0

# cx = 0
# cy = 1

# dx = 1
# dy = 1

# #Input 21
# # No intersection, gap minimum
# ax = 0
# ay = 0

# bx = 1
# by = 0

# cx = 0
# cy = 0.01

# dx = 1
# dy = 0.01

#Input 23
# No intersection, gap minimum
# ax = 0
# ay = 0

# bx = 1
# by = 0

# cx = 0
# cy = 0

# dx = -1
# dy = 0



def get_all_variables(ax, ay, bx, by, cx, cy, dx, dy):
 # m1 m2 m3 m4 ux uy g
 obj = [1,1,1,1,0,0,0]
 column_names = ["m1","m2","m3","m4","ux","uy","g"]
 ub = [cplex.infinity,cplex.infinity,cplex.infinity,cplex.infinity,cplex.infinity,cplex.infinity,cplex.infinity]
 lb = [-cplex.infinity,-cplex.infinity,-cplex.infinity,-cplex.infinity,-cplex.infinity,-cplex.infinity,-cplex.infinity]

 right_side_values = list()
 sense = ""
 rows = list()
 cols = list()
 vals = list()
 rownames = list()

 rownames.append("r1")
 right_side_values.append(0)
 sense = sense+"L"
 rows.append(0)
 cols.append(0)
 vals.append(-1)

 rownames.append("r2")
 right_side_values.append(0)
 sense = sense+"L"
 rows.append(1)
 cols.append(0)
 vals.append(-1)
 rows.append(1)
 cols.append(4)
 vals.append(-ax)
 rows.append(1)
 cols.append(5)
 vals.append(-ay)
 rows.append(1)
 cols.append(6)
 vals.append(-1)

 #add the remaining ones
 rownames.append("r3")
 right_side_values.append(0)
 sense = sense+"L"
 rows.append(2)
 cols.append(1)
 vals.append(-1)

 rownames.append("r4")
 right_side_values.append(0)
 sense = sense+"L"
 rows.append(3)
 cols.append(1)
 vals.append(-1)
 rows.append(3)
 cols.append(4)
 vals.append(-bx)
 rows.append(3)
 cols.append(5)
 vals.append(-by)
 rows.append(3)
 cols.append(6)
 vals.append(-1)

 rownames.append("r5")
 right_side_values.append(0)
 sense = sense+"L"
 rows.append(4)
 cols.append(2)
 vals.append(-1)

 rownames.append("r6")
 right_side_values.append(-1)
 sense = sense+"L"
 rows.append(5)
 cols.append(2)
 vals.append(-1)
 rows.append(5)
 cols.append(4)
 vals.append(cx)
 rows.append(5)
 cols.append(5)
 vals.append(cy)
 rows.append(5)
 cols.append(6)
 vals.append(1)

 rownames.append("r7")
 right_side_values.append(0)
 sense = sense+"L"
 rows.append(6)
 cols.append(3)
 vals.append(-1)

 rownames.append("r8")
 right_side_values.append(-1)
 sense = sense+"L"
 rows.append(7)
 cols.append(3)
 vals.append(-1)
 rows.append(7)
 cols.append(4)
 vals.append(dx)
 rows.append(7)
 cols.append(5)
 vals.append(dy)
 rows.append(7)
 cols.append(6)
 vals.append(1)


 prob = cplex.Cplex()
 prob.objective.set_sense(prob.objective.sense.minimize)
 prob.linear_constraints.add(rhs = right_side_values, senses = sense, names = rownames)
 prob.variables.add(obj = obj, ub = ub, lb = lb, names = column_names)

 name_indices = [i for i in range(len(obj))]
 names = prob.variables.get_names(name_indices)

 prob.set_log_stream(None)
 prob.set_error_stream(None)
 prob.set_warning_stream(None)
 prob.set_results_stream(None)

 prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
 prob.solve()

 #print "Solution value  = ", prob.solution.get_objective_value()
 numcols = prob.variables.get_num()
 x = prob.solution.get_values()
 #for j in range(numcols):
 # print("Column %s:  Value = %10f" % (names[j], x[j]))

 return x, numcols

def get_ux(ax, ay, bx, by, cx, cy, dx, dy):
 x, numcols = get_all_variables(ax, ay, bx, by, cx, cy, dx, dy)
 return x[numcols-3]

def get_uy(ax, ay, bx, by, cx, cy, dx, dy):
 x, numcols = get_all_variables(ax, ay, bx, by, cx, cy, dx, dy)
 return x[numcols-2]

def get_gamma(ax, ay, bx, by, cx, cy, dx, dy):
 x, numcols = get_all_variables(ax, ay, bx, by, cx, cy, dx, dy)
 return x[numcols-1]

def get_u_gamma(ax, ay, bx, by, cx, cy, dx, dy):
 x, numcols = get_all_variables(ax, ay, bx, by, cx, cy, dx, dy)
 return x[numcols-3], x[numcols-2], x[numcols-1]

# to get ux
#print(get_ux(ax, ay, bx, by, cx, cy, dx, dy))
# to get uy
#print(get_uy(ax, ay, bx, by, cx, cy, dx, dy))
# to get gamma
#print(get_gamma(ax, ay, bx, by, cx, cy, dx, dy))
