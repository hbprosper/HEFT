
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy.optimize as op
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# linearly transform a feature to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

def split_data(data,
               test_fraction, 
               validation_fraction):

    # Split data into a part for training and a part for testing
    train_data, test_data = train_test_split(data, 
                                         test_size=test_fraction, 
                                         shuffle=True)

    # Split the training data into a part for training (fitting) and
    # a part for validating the training.
    v_fraction = validation_fraction * len(data) / len(train_data)
    train_data, valid_data = train_test_split(train_data, 
                                          test_size=v_fraction,
                                          shuffle=True)

    # reset the indices in the dataframes and drop the old ones
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)
    
    return train_data, valid_data, test_data 

def split_source_target(df, source, target):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    x = np.array(df[source])
    t = np.array(df[target])
    return x, t

# return a batch of data for the next step in minimization
def get_batch(x, t, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    batch_t = t[rows]
    return (batch_x, batch_t)

# Note: there are several average loss functions available 
# in pytorch, but it's useful to know how to create your own.
def average_quadratic_loss(f, t, x=None):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)

def average_cross_entropy_loss(f, t, x=None):
    # f and t must be of the same shape
    loss = torch.where(t > 0.5, torch.log(f), torch.log(1 - f))
    return -torch.mean(loss)

def average_quantile_loss(f, t, x):
    # f and t must be of the same shape
    tau = x.T[-1] # last column is tau.
    return torch.mean(torch.where(t >= f, 
                                  tau * (t - f), 
                                  (1 - tau)*(f - t)))

# function to validate model during training.
def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float().to(device)
        t = torch.from_numpy(targets).float().to(device)
        # remember to reshape!
        o = model(x).reshape(t.shape)
    return avloss(o, t, x)
        
def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
def train(model, optimizer, dictfile, early_stopping_count,
          avloss, getbatch,
          train_data, valid_data, 
          features, target,
          batch_size,
          n_iterations, 
          traces, 
          step=10, 
          change=0.005):
    
    train_x, train_t = split_source_target(train_data, 
                                           features, target)
    
    valid_x, valid_t = split_source_target(valid_data, 
                                           features, target)
    
    # to keep track of average losses
    xx, yy_t, yy_v = traces

    # place model on current computational device
    model = model.to(device)
    
    # save model with smallest validation loss
    # if after early_stopping_count iterations 
    # no validation scores are lower than the
    # current lowest value.
    min_acc_v = 1.e30
    stopping_count = 0
    jjsaved = 0
    
    n = len(valid_x)
    
    print('Iteration vs average loss')
    print("%9s %9s %9s" % \
          ('iteration', 'train-set', 'valid-set'))
    
    for ii in range(n_iterations):
                
        stopping_count += 1
            
        # set mode to training so that training specific 
        # operations such as dropout are enabled.
        model.train()
        
        # get a random sample (a batch) of data (as numpy arrays)
        batch_x, batch_t = getbatch(train_x, train_t, batch_size)
        
        # convert the numpy arrays batch_x and batch_t to tensor 
        # types. The PyTorch tensor type is the magic that permits 
        # automatic differentiation with respect to parameters. 
        # However, since we do not need to take the derivatives
        # with respect to x and t, we disable this feature
        with torch.no_grad(): # no need to compute gradients 
            # wrt. x and t
            x = torch.from_numpy(batch_x).float().to(device)
            t = torch.from_numpy(batch_t).float().to(device)
 
        # compute the output of the model for the batch of data x
        # Note: outputs is 
        #   of shape (-1, 1), but the tensor targets, t, is
        #   of shape (-1,)
        # for the tensor operations with outputs and t to work
        # correctly, it is necessary that they be of the same
        # shape. We can do this with the reshape method.
        outputs = model(x).reshape(t.shape)
        
        # compute a noisy approximation to the average loss
        empirical_risk = avloss(outputs, t, x)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients
        
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step()            # move one step
        
        if ii % step == 0:
            
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]).item() 
            acc_v = validate(model, avloss, valid_x[:n], valid_t[:n]).item()
            print(f'\r{ii}',end='')

            # save only if there has been a "significant" change in 
            # the average loss.
            if acc_v < (1 - change)*min_acc_v:
                min_acc_v = acc_v
                torch.save(model.state_dict(), dictfile)
                stopping_count = 0
                jjsaved = ii
            else:
                if stopping_count > early_stopping_count:
                    print('\n\nstopping early!')
                    break
                    
            if len(xx) < 1:
                xx.append(0)
                print("%9d %9.7f %9.7f" % (xx[-1], acc_t, acc_v))
            elif len(xx) < 5:
                xx.append(xx[-1] + step)
                print("%9d %9.7f %9.7f" % (xx[-1], acc_t, acc_v))
            else:
                xx.append(xx[-1] + step)
                saved = ' %9d: %9d/%10.8f/%9d' % \
                (ii, jjsaved, min_acc_v, stopping_count)
                print("\r%9d %9.7f %9.7f%s" % \
                      (xx[-1], acc_t, acc_v, saved), end='')
                
            yy_t.append(acc_t)
            yy_v.append(acc_v)
                
    print()

    torch.save(model.state_dict(), dictfile.replace('.', '-final.'))
    return (xx, yy_t, yy_v)

def plot_average_loss(traces, ftsize=18, filename='fig_loss.pdf'):
    
    xx, yy_t, yy_v = traces
    
    # create an empty figure
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')

    ax.set_xlabel('Iterations', fontsize=ftsize)
    ax.set_ylabel('average loss', fontsize=ftsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()