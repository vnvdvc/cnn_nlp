import os,re
from datetime import datetime

from cnn import *


class Discrete(object):
  def __init__(self,name,domain):
  #domain is a set of SORTED values available to the parameter
      self.name = name
      self.domain = domain

class Node(object):
  

  def __init__(self,filters,value):
    self.filters = filters
    self.value = value
    self.children = []
    self.parent = None
    self.complete = False

  def is_sibling(self,node):
    if self.filters[:-1] == node.filters[:-1]:
      return True
    else:
      return False

  def is_child(self,node):
    if self.filters == node.filters[:-1]:
      return True
    else:
      return False

  def next_child(self,node):
    if node.parent.filters == self.filters:
      filters = node.filters
      idx = 0
      for idx in range(len(self.children)):
        if self.children[idx].filters == filters:
          break
      if idx == len(self.children) - 1:
        return None
      else:
        return self.children[idx+1]
    else:
      raise KeyError("Wrong child selected:\nparent: "+str(self.filters) + "\nchild: "+str(node.filter))

  def save_node(self,filename):
    with open(filename,"a+") as inp:
      inp.write(','.join([str(filt) for filt in self.filters])+"\n")
      inp.write(str(self.value)+"\n")
      inp.write(str(int(self.complete))+"\n")
      return

#Check if one node is the ancestor of the other node, if True return the child
  def is_ancestor(self,node):
    if len(node.filters) < len(self.filters):
      small = self
      big = node
    elif len(node.filters) > len(self.filters):
      small = node
      big = self
    else:
      if node.filters == self.filters:
        raise KeyError("Fatal: Same node visited twice!\n"+"filters: {}".format(self.filters))
      else:
        return False
    i = 1
    while i <= len(small.filters):
      if big.filters == small.filters[:-i]:
        return small
      i += 1
    return False







class Opt(object):
  
  def __init__(self,config,Pars,special,results_file):
    """
    Pars: parameters
    special: special parameters like config.filters
    """
    self.Pars = Pars
    self.special = special
    
    self.best_config = config

    self.output = results_file
    

  def single_cal(self,config):
    """
    wrapper for run_cnn()
    return: val_acc
    """
    return run_cnn(config,self.output)


  def save_filter_tree(self,root,good_nodes):
    """
    function used to save the computed tree to a file
    good_nodes are saved at the top, one filter a line.
    The tree is saved from left branch to right branch, depth first, with root denoted as a line "root\n" at the beginning.
    Each node has three lines:
    line 1: node.filters, line 2: node.value, line 3: node.complete, 0 is false and 1 is true.
    """
    date = str(datetime.now())[5:10]
    filename = "./log/filter_tree_" + date
    if os.path.isfile(filename):
      os.remove(filename)
    if good_nodes == []:
      with open(filename,"a+") as inp:
        inp.write("root\n")
    else:
      for node in good_nodes:
        node.save_node(filename)
      with open(filename,"a+") as inp:
        inp.write("root\n")
    
    parent = root
    curr_node = root.children[0]
    while True:
      curr_node.save_node(filename)
      if curr_node.children == []:
        while parent.next_child(curr_node) is None:
          if parent == root and parent.next_child(curr_node) is None:
            return
          curr_node = parent
          parent = curr_node.parent
        curr_node = parent.next_child(curr_node)
      else:
        parent = curr_node
        curr_node = parent.children[0]
        
        
     

    
  def read_filter_tree(self,filename):
    """
    function used to read a previously saved tree, and good nodes
    return root, good_nodes
    """
    root = Node([],0.0)
    curr_node = root
    good_nodes = []
    with open(filename,"r") as inp:
      line = inp.readline()
      while line != "root\n":
        if line.isspace():
          line = inp.readline()
        if line[-1] == "\n":
          line = line[:-1]
        filters = [int(filt) for filt in re.split(',+',line)]
        line = inp.readline()
        value = float(line[:-1])
        line = inp.readline()
        complete = bool(line[:-1])
        node = Node(filters,value)
        node.complete = complete
        good_nodes.append(node)
        line = inp.readline()

      line = inp.readline()
      while line != '':
        if line.isspace():
          line = inp.readline()
        if line[-1] == "\n":
          line = line[:-1]
        filters = [int(filt) for filt in re.split(',+',line)]
        line = inp.readline()
        value = float(line[:-1])
        line = inp.readline()
        complete = bool(line[:-1])
        node = Node(filters,value)
        node.complete = complete
        if curr_node.is_child(node):
          curr_node.children.append(node)
          parent = curr_node
          curr_node = node
          curr_node.parent = parent
        elif curr_node.is_sibling(node):
          parent.children.append(node)
          curr_node = node
          curr_node.parent = parent
        else:
          while parent.is_child(node) is False:
            if parent == root:
              raise KeyError("No parent found for the node: \n"+str(node.filters)+"\n")
            parent = parent.parent

          parent.children.append(node)
          curr_node = node
          curr_node.parent = parent

        line = inp.readline()
      return
          



        
         

        
  

  def optimal_filters(self,treefile=None):
    """
    optimize different sizes of filters: the state space can be treated as a tree, with all single filters, that is
    [1],[2],etc, at the first layer, and double filters at second layer, so on and so forth. Each node is a set of filters. Every node containing the filter size "1" has the common ancestor [1], and every node containing "2" has ancestor [2] except those containing "1".
    In this algorithm, it is assumed that if adding a filter size does not immediately improve the result, all children of the new filter 
do not improve the result either.
    """
    config = self.best_config
    name = self.special.name
    domain = self.special.domain
    num_of_nodes = 0
    checkpt = 30
    if treefile is not None:
      root,good_nodes = self.read_filter_tree(treefile)
      curr_node = root
      while curr_node.complete is False:
        for node in curr_node.children:
          if node.complete is False:
            curr_node = node
            break
        if curr_node.children == []:
          break
      for filter_idx in range(len(domain)):
        if domain[filter_idx] == temp_filter[-1]:
          break
      temp_filter = curr_node.filters + [domain[filter_idx+1]]

    else:
      root = Node([],0.0)
      good_nodes = [] #A good node is defined as that, along the path connecting root to the node, each node's value is smaller than its child's value.
      curr_node = root
      temp_filter = domain[:1]

    while True:
      config.filters = temp_filter
      temp_acc = self.single_cal(config)
      newnode = Node(temp_filter,temp_acc)
      if curr_node.is_child(newnode):
          curr_node.children.append(newnode)
          newnode.parent = curr_node
      else:
          print "Error!:\n" + "child: " + str(newnode.filters) +"\nparent: " + str(curr_node.filters)
          curr_node.parent.children.append(newnode)
          newnode.parent = curr_node.parent
      num_of_nodes += 1
      if num_of_nodes % checkpt == 0:
        self.save_filter_tree(root,good_nodes)
      for filter_idx in range(len(domain)):
        if domain[filter_idx] == temp_filter[-1]:
          break
#check if newnode is the leaf 
      if filter_idx == len(domain) - 1:
        newnode.complete = True
        newnode.parent.complete = True
        if newnode.value > newnode.parent.value:
          curr_node = newnode
        else:
          curr_node = newnode.parent
        temporary_counter = 0
        temporary_index = 0
        while temporary_index < len(good_nodes):
          node = good_nodes[temporary_index]
          small =  node.is_ancestor(newnode)
          if small is False:
            temporary_index += 1
            continue
          else:
            temporary_counter = 1
            break
        if temporary_counter == 0:
          good_nodes.append(curr_node)
        else:
          good_nodes[temporary_index] = small

        print "good_nodes:\n"
        for node in good_nodes:
            print str(node.filters)

        while curr_node.complete: 
          while curr_node.complete:
            if curr_node.filters == root.filters:
              print "Optimization completed!"
              print "# of nodes calculated: {}".format(num_of_nodes)
              return
            curr_node = curr_node.parent
          
          print curr_node.filters
          temp_filter = curr_node.children[-1].filters
          for filter_idx in range(len(domain)):
            if domain[filter_idx] == temp_filter[-1]:
              break
          temp_filter = curr_node.filters + [domain[filter_idx+1]]
#If temp_filter is a subset of a "good" filter, which comes from a good node, the optimization will be constrained within the good filter, based on Assumption 2.

          if good_nodes != []:
            for node in good_nodes:
              for filt in temp_filter:
                if filt not in node.filters:
                  break
                if filt == temp_filter[-1]:
                  idx = 0
                  while filt != node.filters[idx]:
                    idx += 1
                  if idx == len(temp_filter) - 1:
                    break
                  else:
                    local_domain = node.filters
                    local_curr_node = curr_node
                    while idx < len(local_domain):
                      config.filters = temp_filter
                      temp_acc = self.single_cal(config)
                      newnode = Node(temp_filter,temp_acc)
                      newnode.complete = True
                      local_curr_node.children.append(newnode)
                      newnode.parent = local_curr_node
                      num_of_nodes += 1
                      if num_of_nodes % checkpt == 0:
                        self.save_filter_tree(root,good_nodes)
                      if temp_acc > curr_node.value:
                        local_curr_node = newnode
                      else:
                        local_curr_node = newnode.parent
                      if idx == len(local_domain) -1:
                        break
                      else:
                        temp_filter = local_curr_node.filters + [local_domain[idx+1]]
                      idx += 1
                    if curr_node.children[-1].filters[-1] == domain[-1]:
                      curr_node.complete = True
                    else:
                      curr_node = curr_node.children[-1]

          else:
            break

                  
                

      else:
        if newnode.value > newnode.parent.value:
          curr_node = newnode
          
        else:
          curr_node.complete = True
          curr_node = newnode.parent

        temp_filter = curr_node.filters+[domain[filter_idx+1]]

      if root.children[-1].filters == domain[-1:]:
         print "Optimization completed!"
         print "# of nodes calculated: {}".format(num_of_nodes)
         return

  def full_tree(self,treefile=None):
    config = self.best_config
    domain = self.special.domain
    num_of_nodes = 0
    checkpt = 30
    if treefile is not None:
      root,good_nodes = self.read_filter_tree(treefile)
      curr_node = root
      while curr_node.complete is False:
        for node in curr_node.children:
          if node.complete is False:
            curr_node = node
            break
        if curr_node.children == []:
          break
      for filter_idx in range(len(domain)):
        if domain[filter_idx] == temp_filter[-1]:
          break
      temp_filter = curr_node.filters + [domain[filter_idx+1]]

    else:
      root = Node([],0.0)
      curr_node = root
      temp_filter = domain[:1]
      good_nodes = []

    while True:
      config.filters = temp_filter
      temp_acc = self.single_cal(config)
      newnode = Node(temp_filter,temp_acc)
      if curr_node.is_child(newnode):
          curr_node.children.append(newnode)
          newnode.parent = curr_node
      else:
          print "Error!:\n" + "child: " + str(newnode.filters) +"\nparent: " + str(curr_node.filters)
          return
      num_of_nodes += 1
      if num_of_nodes % checkpt == 0:
        self.save_filter_tree(root,good_nodes)
      for filter_idx in range(len(domain)):
        if domain[filter_idx] == temp_filter[-1]:
          break
#check if newnode is the leaf 
      if filter_idx == len(domain) - 1:
        newnode.complete = True
        curr_node.complete = True
        while curr_node.complete:
          if curr_node.filters == root.filters:
            print "Optimization completed!"
            print "Calculated nodes: {}".format(num_of_nodes)
            return
          curr_node = curr_node.parent

        print curr_node.filters
        temp_filter = curr_node.children[-1].filters
        for filter_idx in range(len(domain)):
          if domain[filter_idx] == temp_filter[-1]:
            break
        temp_filter = curr_node.filters + [domain[filter_idx+1]]

      else:
        curr_node = newnode
        temp_filter = curr_node.filters + [domain[filter_idx+1]]



if __name__ == "__main__":
  now = str(datetime.now())[5:10]
  output = "./log/results_" + now + "_0.txt"
  while os.path.isfile(output):
    output = output[:-4] + "0.txt"

  config = Config()
  domain = [1,2,3,4,5,6,7,8,9,10,15]
  filters = Discrete("filters",domain)
  
  opt = Opt(config,None,filters,output)
  # opt.optimal_filters()
  opt.full_tree()
  
