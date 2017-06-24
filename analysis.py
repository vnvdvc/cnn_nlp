import matplotlib.pyplot as plt
import re,sys
import numpy as np

class Model_params(object):
  
  label_size =3
  sentence_length =140
  max_epoch = 100
  early_stopping = 2
  real_analysis = None

  def __init__(self,params_str):
     linenum = 0 
     while linenum < len(params_str): 
          line = params_str[linenum]
          if line[-1] == '\n':
             line = line[:-1]
          if line[:7] == 'Results':
             linenum += 1
             line = params_str[linenum]

#find embed_size and batch_size
             temp = re.findall('\d+',line)
             assert len(temp) == 2, "Embed and batch, wrong line is read:\n" + line 
             self.embed_size = int(temp[0])
             self.batch_size = int(temp[1])
#find activation, relu or tanh
             t_idx = line.find('-')
             temp = line[t_idx+1:t_idx+5]
             assert temp in ['relu','tanh'], "Wrong activation:\n" + line
             self.activation = temp
#find filters
             linenum += 1
             line = params_str[linenum]
             temp = re.findall('\d+',line)
             assert int(temp[0]) in [1,2], "Wrong # of layers:\n" + line
             if temp[0] == '1':
                self.num_of_filters = int(temp[1])
                self.filters = [int(item) for item in temp[2:]]
             else:
                t_idx = line.find('hidden')
                hid = line[t_idx+7:t_idx+11]
                if hid in ['tanh','relu']:
                    self.activation_hidden = hid
                elif line[t_idx+7:t_idx+14] == 'sigmoid':
                    self.activation_hidden = 'sigmoid'
                else:
                    raise TypeError('Wrong activation_hidden used')
#find lr and l2
                self.hidden_size = int(temp[1])
                self.num_of_filters = int(temp[2])
                self.filters = [int(item) for item in temp[3:]] 

                linenum += 1
                line = params_str[linenum]
                temp = re.findall('0+[.]\d+',line)
                assert len(temp) == 3, "Wrong rates:\n" + line
                self.lr = float(temp[0])
                self.l2_conv = float(temp[1])
                self.l2_softmax = float(temp[2])
#find dropout  
                linenum += 1
                line = params_str[linenum]
                temp = re.findall('0+[.]\d+',line)
                assert len(temp) == 1, "Wrong dropout:\n" + line
                self.dropout = float(temp[0]) 
                return 

class Analysis(object):
  
  def __init__(self,inp_file):
    
      self.train_losses, self.val_losses = self.get_losses(inp_file) 
      self.train_accs, self.val_accs = self.get_accuracies(inp_file)
      self.converged_t_accs, self.converged_v_accs = self.final_accs()
      self.big10 = self.big10_accs()


  def get_losses(self,inp_file):
      inp = open(inp_file,"r")
      train_losses = []
      val_losses = []
      line = inp.readline()
      while line:
        if line == "Losses of train and val datasets:\n":
            temp = inp.readline()
            if temp[-1] == '\n':
                temp = temp[:-1]
            train_losses.append([float(num) for num in temp.split(',')])
            temp = inp.readline()
            if temp[-1] == '\n':
                temp = temp[:-1]
            val_losses.append([float(num) for num in temp.split(',')])
  
        line = inp.readline()
  
      return np.array(train_losses),np.array(val_losses)
      
  
  def get_accuracies(self,inp_file):
      inp = open(inp_file,"r")
      train_accuracies = []
      val_accuracies = []
      line = inp.readline()
      while line:
        if line == "Accuracies of train and val datasets:\n":
            temp = inp.readline()
            if temp[-1] == '\n':
                temp = temp[:-1]
            train_accuracies.append([float(num) for num in temp.split(',')])
            temp = inp.readline()
            if temp[-1] == '\n':
                temp = temp[:-1]
            val_accuracies.append([float(num) for num in temp.split(',')])
  
        line = inp.readline()

      return np.array(train_accuracies),np.array(val_accuracies)
      
  
  def final_accs(self):
      final_train = []
      final_val = []
      for train_trial,val_trial in zip(self.train_accs,self.val_accs):
          final_train.append(train_trial[-1])
          final_val.append(val_trial[-1])
      
      return np.array(final_train),np.array(final_val)
      
  # return a list of tuples,(ind,train_acc,val_acc) in the ascending order of the val_acc
  def big10_accs(self,selected_samples=10):
      val_tuples = list(enumerate(self.converged_v_accs))
      if len(val_tuples) > 10:
        big10_accs = sorted(val_tuples,key=lambda val: val[1])[-selected_samples:]
      else:
        big10_accs = sorted(val_tuples,key=lambda val: val[1])

      results = []
      for ind,val_acc in big10_accs:
          results.append((ind,self.converged_t_accs[ind],val_acc))
  
      return results
  
  def graph_big10(self,selected_samples=10): 
      
      sample_idxs = [item[0] for item in self.big10] 
      t_accs = self.train_accs[sample_idxs]
      v_accs = self.val_accs[sample_idxs]
      t_losses = self.train_losses[sample_idxs]
      v_losses = self.val_losses[sample_idxs]

      fig = plt.figure() 
      t_ax = fig.add_subplot(221)
      v_ax = fig.add_subplot(222)
      tl_ax = fig.add_subplot(223)
      vl_ax = fig.add_subplot(224)

      t_ax.set_title('train accuracies')
      v_ax.set_title('val accuracies')
      tl_ax.set_title('train losses')
      vl_ax.set_title('val losses')

      t_ax.set_ylabel('accuracies')
      tl_ax.set_ylabel('losses')
    
      tl_ax.set_xlabel('epoch')
      vl_ax.set_xlabel('epoch')


#fixing the scale of x
      for idx in range(selected_samples):
        y1 = t_accs[idx]
        x1 = np.arange(len(y1))/float(len(y1))
        t_ax.plot(x1,y1,label=str(sample_idxs[idx]))
        y2 = v_accs[idx]
        x2 = np.arange(len(y2))/float(len(y2))
        v_ax.plot(x2,y2)
        y3 = t_losses[idx]
        x3 = np.arange(len(y3))/float(len(y3))
        tl_ax.plot(x3,y3)
        y4 = v_losses[idx]
        x4 = np.arange(len(y4))/float(len(y4))
        vl_ax.plot(x4,y4)

      t_ax.legend(loc=4)
      plt.show()
      return



#idx is the listed order of the result needed in the results file
def get_params(idx,inp_file):
    params_str = []
    count = 0
    with open(inp_file,"r") as inp:
        line = inp.readline()
        while line:
            if line[:7] == 'Results':
               if count == idx:
# lines are put into the list of param_strings
                  total = 5
                  while total > 0:
                    params_str.append(line)
                    line = inp.readline()
                    total -= 1
                  model = Model_params(params_str)
                  return model
               else:
                  count += 1
            line = inp.readline()
             
    raise TypeError('Wrong index of trained model used.') 

def print_params(idx,inp_file):
      count = 0
      with open(inp_file,"r") as inp:
           line = inp.readline()
           while line:
              if line[:7] == 'Results':
                 if count == idx:
#5 lines are put into the list of param_strings
                    total = 4
                    print "Model index in file " + inp_file + ":{}\n".format(idx)
                    while total > 0:
                      line = inp.readline()
                      total -= 1
                      print line
                    total = 3
                    while line[0:8] != "postive,":
                      line = inp.readline()
                    while total>0:
                      print line
                      line = inp.readline()
                      total -= 1
                    while line[0:17] != "Accuracy of Test:":
                      line = inp.readline()
                    print line
                    print '\n'
                    return
                 else:
                    count += 1
              line = inp.readline()
               
      raise TypeError('Wrong index of trained model used.') 

def save_params(idx,inp_file,output):
      count = 0
      with open(inp_file,"r") as inp:
           line = inp.readline()
           while line:
              if line[:7] == 'Results':
                 if count == idx:
#5 lines are put into the list of param_strings
                    total = 4
                    print "Model index in file " + inp_file + ":{}\n".format(idx)
                    while total > 0:
                      line = inp.readline()
                      total -= 1
                      print line
                    total = 3
                    while line[0:8] != "postive,":
                      line = inp.readline()
                    while total>0:
                      print line
                      line = inp.readline()
                      total -= 1
                    while line[0:17] != "Accuracy of Test:":
                      line = inp.readline()
                    print line
                    print '\n'
                    return
                 else:
                    count += 1
              line = inp.readline()
               
      raise TypeError('Wrong index of trained model used.') 

def main(inp):
  ana = Analysis(inp)
  for trial in ana.big10:
    print_params(trial[0],inp)
  ana.graph_big10()

if __name__ == "__main__":
  main(sys.argv[1])



        
        






                


                
