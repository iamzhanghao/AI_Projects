# Written by Patricia Suriana, MIT ca. 2013
# Modified by Tomas Lozano-Perez, MIT ca 2016

import pddl_parser
import search
import time
import sys
import pdb

def printOutputVerbose(tic, toc, path, cost, final_state, goal):
    print "\n******************************FINISHED TEST******************************"
    print "Goals: "
    for state in goal:
      print "\t" + str(state)
    print '\nRunning time: ', (toc-tic), 's'
    if path == None:
        print '\tNO PATH FOUND'
    else:
        print "\nNumber of Actions: ", len(path)
        print '\nCost:', cost
        print "\nPath: "
        for op in path:
          print "\t" + repr(op)
        print "\nFinal States:"
        for state in final_state:
          print "\t" + str(state)
    print "*************************************************************************\n"

def printOutput(tic, toc, path, cost):
    print (toc-tic), '\t', len(path), '\t', cost          

if __name__ == "__main__":
  args = sys.argv
  if len(args) != 3:                    # default task
      dirName = "prodigy-bw"
      fileName = "bw-simple"
  else:
      dirName = args[1]
      fileName = args[2]
  domain_file = dirName + '/domain.pddl'
  problem_file = dirName + '/' + fileName + '.pddl'

  # task is an instance of the Task class defined in strips.py
  task = pddl_parser.parse(domain_file, problem_file)

  # This should be commented out for larger tasks
  print task

  print "\n******************************START TEST******************************"
  
  tic = time.time()

  # Define a sub-class of the Problem class, make an instance for the task and call the search
  # You should then set the variables:
  # final_state - the final state at the end of the plan
  # plan - a list of actions representing the plan
  # cost - the cost of the plan
  # Your code here

  toc = time.time()
  printOutputVerbose(tic, toc, plan, cost, final_state, task.goals)

  
