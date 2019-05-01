# from __future__ import print_function
# from ortools.linear_solver import pywraplp
# import ortools
# print(ortools.__version__)

# def main():
#   solver = pywraplp.Solver('SolveSimpleSystem',
#                            pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
#   # Create the variables x and y.
#   x = solver.NumVar(0, 1, 'x')
#   y = solver.NumVar(0, 2, 'y')
#   # Create the objective function, x + y.
#   objective = solver.Objective()
#   objective.SetCoefficient(x, 1)
#   objective.SetCoefficient(y, 1)
#   objective.SetMaximization()
#   # Call the solver and display the results.
#   solver.Solve()
#   print('Solution:')
#   print('x = ', x.solution_value())
#   print('y = ', y.solution_value())

# if __name__ == '__main__':
#   main()

"""Linear optimization example"""

from __future__ import print_function
from ortools.linear_solver import pywraplp

def main():
  # Instantiate a Glop solver, naming it LinearExample.
  solver = pywraplp.Solver('LinearExample',
                           pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

# Create the two variables and let them take on any value.
  x = solver.NumVar(-solver.infinity(), solver.infinity(), 'x')
  y = solver.NumVar(-solver.infinity(), solver.infinity(), 'y')

  # Constraint 1: x + 2y <= 14.
  constraint1 = solver.Constraint(-solver.infinity(), 14)
  constraint1.SetCoefficient(x, 1)
  constraint1.SetCoefficient(y, 2)

  # Constraint 2: 3x - y >= 0.
  constraint2 = solver.Constraint(0, solver.infinity())
  constraint2.SetCoefficient(x, 3)
  constraint2.SetCoefficient(y, -1)

  # Constraint 3: x - y <= 2.
  constraint3 = solver.Constraint(-solver.infinity(), 2)
  constraint3.SetCoefficient(x, 1)
  constraint3.SetCoefficient(y, -1)

  # Objective function: 3x + 4y.
  objective = solver.Objective()
  objective.SetCoefficient(x, 3)
  objective.SetCoefficient(y, 4)
  objective.SetMaximization()

  # Solve the system.
  solver.Solve()
  opt_solution = 3 * x.solution_value() + 4 * y.solution_value()
  print('Number of variables =', solver.NumVariables())
  print('Number of constraints =', solver.NumConstraints())
  # The value of each variable in the solution.
  print('Solution:')
  print('x = ', x.solution_value())
  print('y = ', y.solution_value())
  # The objective value of the solution.
  print('Optimal objective value =', opt_solution)

def main1(variable_list, constraint_list):
  solver = pywraplp.Solver('LinearExample',
                           pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

  numvar_list = []
  for var in variable_list:
    numvar = solver.NumVar(0.0, 1.0, var[0])
    numvar_list.append(numvar)

  constraint1 = solver.Constraint(1.0, 1.0)
  for numvar in numvar_list:
    constraint1.SetCoefficient(numvar, 1)

  solver_constraints = [constraint1]
  for cons in constraint_list:
    solver_constraint = solver.Constraint(cons[0], cons[0])
    cons_name = cons[1]
    for i in range(len(variable_list)):
      var = variable_list[i]
      if cons_name in var[1]:
        solver_constraint.SetCoefficient(numvar_list[i], 1)
    solver_constraints.append(solver_constraint)

  print(numvar_list)
  print(solver_constraints)

  objective = solver.Objective()
  for i in range(len(variable_list)):
    var = variable_list[i]
    objective.SetCoefficient(numvar_list[i], len(var[1]))
  objective.SetMaximization()

  solver.Solve()

  print('Solution:')
  for i in range(len(variable_list)):
    var = variable_list[i]
    print(var[0]+' = ', numvar_list[i].solution_value())

  return

if __name__ == '__main__':
  # python 3.6

  p1 = ('p1', ["(A,A')", "(B,B')"]) # (A,A'), (B,B')
  p2 = ('p2', ["(A,A')"])           # (A,A')
  p3 = ('p3', ["(B,B')"])           # (B,B')
  p4 = ('p4', [])                   # empty

  c1 = (0.6, "(A,A')") # p(A,A') = 0.6
  c2 = (0.5, "(B,B')") # p(B,B') = 0.5

  variable_list = [p1,p2,p3,p4]
  constraint_list = [c1,c2]

  # main()
  main1(variable_list, constraint_list)
