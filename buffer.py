from collections import deque
import random
from utilities import transpose_list

# Replay buffer implementation: 
# It implements a size-limited storage of data tuples, from which random samples can be requested. 

class ReplayBuffer:

    #initialize the Replay buffer to store maximum [size] elements
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    # push a data tuple (transition) to the end of the storage
    def push(self,transition):
        """push into the buffer"""        
        input_to_buffer = transpose_list(transition)
        self.deque.append(transition)

    # get [batchsize] elements randomly from the memory buffer
    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)
        return samples

    # overwrite len function to return the actually used-up buffer size 
    def __len__(self):
        return len(self.deque)



