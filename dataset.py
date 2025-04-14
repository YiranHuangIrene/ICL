import numpy as np
import torch
from torch.utils.data import IterableDataset

def get_mus_label_class(K, L, D):
    """Generate label and class embeddings"""
    mus_label = np.random.normal(size=(L, D))/np.sqrt(D)
    mus_class = np.random.normal(size=(K, D))/np.sqrt(D)
    
    if K < L or K%L != 0:
        raise ValueError("K > L and K%L == 0 is required")
        
    labels_class = np.tile(np.arange(L), int(K/L))
    
    return mus_label, mus_class, labels_class

def generate_input_seqs(mus_label, mus_class, labels_class, N,S, Nmax, eps= 0.1, B = 0, p_B = 0, P = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, seq_labels = False, rope = True):
    e_fac = 1/np.sqrt(1+eps**2)
    
    L = mus_label.shape[0]
    K = mus_class.shape[0]
    D = mus_label.shape[1]


    K_c = 128
    mus_class_new = np.random.normal(size = (K_c,D))/np.sqrt(D)

    if K_c < L or K_c%L != 0:
        print("K > L and K%L == 0 is required")
        return 0
    labels_class_new =  np.tile(np.arange(L),int(K_c/L))
    
    if rope:
        input_dim = D
    else:
        input_dim = 2*Nmax+1 + D
    inputs = np.zeros((S,2*N+1,input_dim))

    if P is None or len(P) != K:
        P = np.ones(K)/K

    #N has to be divisible by B as we will have N/B copies of each label in the context. 
    if (B > 0 and N%B != 0) or B >= N:
        print("N is not divisible by B or N/B is not even or B >= N")
        return 0,0

    if B == 0:
        B = int(N/2)
        p_B = 0

    choices = np.zeros((S,int(N/B)), dtype = int)
    if no_repeats:
        for s in range(S):
            label_choices = np.random.choice(np.arange(L), size = (int(N/B)), replace = False)
            pos_choices = np.random.choice(np.arange(int(K/L)), size = (int(N/B)))
            choices[s] = pos_choices*L + label_choices
    else:
        choices = np.random.choice(np.arange(K), size = (S,int(N/B)), p=P)
    choices = np.tile(choices,B)
    [np.random.shuffle(x) for x in choices]

    choices_c = np.zeros((S,int(N/B)), dtype = int)
    if no_repeats:
        for s in range(S):
            label_choices = np.random.choice(np.arange(L), size = (int(N/B)), replace = False)
            pos_choices = np.random.choice(np.arange(int(K_c/L)), size = (int(N/B)))
            choices_c[s] = pos_choices*L + label_choices
            #print(choices_c[s], labels_class_new[choices_c[s]],label_choices, pos_choices)
    else:
        choices_c = np.random.choice(np.arange(K_c), size = (S,int(N/B)))
    choices_c = np.tile(choices_c,B)
    [np.random.shuffle(x) for x in choices_c]

    targets_ind = np.random.choice(choices.shape[1], size = (choices.shape[0],)) # select the target from the context
    targets = choices[np.arange(choices.shape[0]),targets_ind]

    targets_c_ind = np.random.choice(choices_c.shape[1], size = (choices_c.shape[0],))
    targets_c = choices_c[np.arange(choices_c.shape[0]),targets_c_ind]

    filt_B = np.random.uniform(size = S) > p_B

    choices[filt_B] = np.random.choice(K,size  = (np.sum(filt_B),N), p = P)
    targets[filt_B] = np.random.choice(K,size  = (np.sum(filt_B),), p = P)

    filt_C = np.random.uniform(size = S) > p_C

    #print(np.arange(S)[~filt_C])
    #print(np.arange(S)[~filt_B])
    if rope:
        start_ind = 0
    else:
        start_ind = 2*Nmax+1
    inputs[filt_C,:-1:2,start_ind:] = (e_fac*(mus_class[choices] + eps*np.random.normal(size = (S,N,D))/np.sqrt(D)))[filt_C]  # Context item embeddings sampled from the bursty sequences
    
    if flip_labels:
        wrong_label = (labels_class + 1)%L
        inputs[filt_C,1:-1:2,start_ind:] = ((mus_label[wrong_label])[choices])[filt_C]
    else:
        inputs[filt_C,1:-1:2,start_ind:] = ((mus_label[labels_class])[choices])[filt_C]  # Label item embeddings sampled from the bursty sequences

    inputs[filt_C,-1,start_ind:] = ((e_fac*(mus_class[targets] + eps*np.random.normal(size = (S,D))/np.sqrt(D))))[filt_C]  # Target item embedding sampled from the bursty sequences

    inputs[~filt_C,:-1:2,start_ind:] = (e_fac*(mus_class_new[choices_c] + eps*np.random.normal(size = (S,N,D))/np.sqrt(D)))[~filt_C] # Context item embeddings sampled from the OOD sequences
    inputs[~filt_C,1:-1:2,start_ind:] = ((mus_label[labels_class_new])[choices_c])[~filt_C]  # Label item embeddings sampled from the OOD sequences
    inputs[~filt_C,-1,start_ind:] = (e_fac*(mus_class_new[targets_c] + eps*np.random.normal(size = (S,D))/np.sqrt(D)))[~filt_C]   # Target item embedding sampled from the OOD sequences

    shifts = np.random.choice((2*Nmax + 1) - (2*N + 1) + 1, size = (S))
    
    labels = np.zeros((S,L),dtype= bool)
    target_classes = np.zeros(S, dtype = int)
    label_sequences = np.zeros((S, N+1), dtype=int)

    for s in range(S):
        if filt_C[s]:
            labels[s,labels_class[targets[s]]] = True
            target_classes[s]= targets[s]
            label_sequences[s] = np.append(labels_class[choices[s]],labels_class[targets[s]])
             
        else:
            labels[s,labels_class_new[targets_c[s]]] = True
            target_classes[s] = -1
            label_sequences[s] = np.append(labels_class_new[choices_c[s]],labels_class_new[targets_c[s]])
        if not rope:
            inputs[s,:,shifts[s]:shifts[s] + 2*N+1] = np.identity(2*N+1)

    if seq_labels:
        return torch.FloatTensor(inputs), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
    elif output_target_labels:
        return torch.FloatTensor(inputs), torch.FloatTensor(labels), torch.LongTensor(target_classes)
    else:
        return torch.FloatTensor(inputs), torch.FloatTensor(labels)

class ICLDataset(IterableDataset):
    def __init__(
        self,
        mus_label: np.ndarray,
        mus_class: np.ndarray,
        labels_class: np.ndarray,
        N: int,  # Number of item-label pairs in context
        S: int,  # Number of sequences
        Nmax: int,  # Maximum context length
        eps: float = 0.1,  # Within-class variance
        B: int = 0,  # Burstiness 
        p_B: float = 0,  # Fraction of bursty sequences
        p_C: float = 0,  # Fraction of OOD sequences
        P = None,  # Class probability distribution
        datasize: int = 10,  # Number of sequences to generate
        flip_labels: bool = False,  # Whether to flip labels
        no_repeats: bool = False,  # Whether to allow repeated items
        rope: bool = False,  # Whether using rotary position embeddings
        output_target_labels: bool = False,  # Whether to output target labels
        seq_labels: bool = False,  # Whether to output sequence labels
    ):
        super().__init__()
        self.L = mus_label.shape[0]
        self.K = mus_class.shape[0]
        self.D = mus_label.shape[1]
        self.mus_label = mus_label
        self.mus_class = mus_class
        self.labels_class = labels_class
        self.N = N
        self.S = S
        self.Nmax = Nmax
        self.eps = eps
        self.B = B
        self.p_B = p_B
        self.p_C = p_C
        self.no_repeats = no_repeats
        self.rope = rope
        self.flip_labels = flip_labels
        self.output_target_labels = output_target_labels
        self.seq_labels = seq_labels
        self.datasize = datasize
        self.n_samples = 0
        
        # Set up class distribution if not provided
        if P is None or len(P) != self.K:
            self.P = np.ones(self.K)/self.K
        else:
            self.P = P
            
        # Set up OOD class embeddings
        self.K_c = 128
        self.mus_class_new = None
        self.labels_class_new = None
        # Validate burstiness parameter
        if B > 0 and N % B != 0:
            raise ValueError("N must be divisible by B when B > 0")
        if B >= N:
            raise ValueError("B must be less than N")
        if B == 0:
            self.B = int(N/2)
            self.p_B = 0

    def generate_sequence(self):
        """Generate a single sequence"""
        e_fac = 1/np.sqrt(1 + self.eps**2)
        
        if self.rope:
            input_dim = self.D
        else:
            input_dim = 2*self.Nmax + 1 + self.D
            
        inputs = np.zeros((self.S, 2*self.N + 1, input_dim))
        
        # Generate context items
        n_bursts = int(self.N/self.B)
        choices = np.zeros((self.S, n_bursts), dtype=int)
        if self.no_repeats:
            for s in range(self.S):
                label_choices = np.random.choice(np.arange(self.L), size=n_bursts, replace=False)
                pos_choices = np.random.choice(np.arange(int(self.K/self.L)), size=n_bursts)
                choices[s] = pos_choices*self.L + label_choices
        else:
            choices = np.random.choice(np.arange(self.K), size=(self.S, n_bursts), p=self.P)
        
        choices = np.tile(choices, self.B)
        [np.random.shuffle(x) for x in choices]
        
        # Generate OOD context items
        self.mus_class_new = np.random.normal(size=(self.K_c, self.D))/np.sqrt(self.D)
        self.labels_class_new = np.tile(np.arange(self.L), int(self.K_c/self.L))
        choices_c = np.zeros((self.S, n_bursts), dtype=int)
        if self.no_repeats:
            for s in range(self.S):
                label_choices = np.random.choice(np.arange(self.L), size=n_bursts, replace=False)
                pos_choices = np.random.choice(np.arange(int(self.K_c/self.L)), size=n_bursts)
                choices_c[s] = pos_choices*self.L + label_choices
        else:
            choices_c = np.random.choice(np.arange(self.K_c), size=(self.S, n_bursts))
            
        choices_c = np.tile(choices_c, self.B)
        [np.random.shuffle(x) for x in choices_c]
        
        # Select target
        target_ind = np.random.choice(choices.shape[1], size=(choices.shape[0],))
        target = choices[np.arange(choices.shape[0]), target_ind]
        
        target_c_ind = np.random.choice(choices_c.shape[1], size=(choices_c.shape[0],))
        target_c = choices_c[np.arange(choices_c.shape[0]), target_c_ind]
        
        # Determine if sequence is bursty and/or OOD
        is_bursty = np.random.uniform(size=self.S) <= self.p_B
        is_ood = np.random.uniform(size=self.S) <= self.p_C
        
        choices[~is_bursty] = np.random.choice(self.K, size=(np.sum(~is_bursty), self.N), p=self.P)
        target[~is_bursty] = np.random.choice(self.K, size=(np.sum(~is_bursty),), p=self.P)
            
        if self.rope:
            start_ind = 0
        else:
            start_ind = 2*self.Nmax + 1
        # Context item embeddings sampled from the existing classes
        inputs[~is_ood, :-1:2, start_ind:] =(e_fac*(self.mus_class[choices] + 
                self.eps*np.random.normal(size=(self.S,self.N, self.D))/np.sqrt(self.D)))[~is_ood]  
        # Label item embeddings sampled from the existing classes with wrong/correct labels
        if self.flip_labels:
            wrong_label = (self.labels_class + 1)%self.L
            inputs[~is_ood, 1:-1:2, start_ind:] = ((self.mus_label[wrong_label])[choices])[~is_ood] 
        else:
            inputs[~is_ood, 1:-1:2, start_ind:] = ((self.mus_label[self.labels_class])[choices])[~is_ood] 
        # Target item embedding sampled from the existing classes
        inputs[~is_ood, -1, start_ind:] = ((e_fac*(self.mus_class[target] + 
                self.eps*np.random.normal(size=(self.S,self.D))/np.sqrt(self.D)))[~is_ood]) 
        
        # Context item embeddings sampled from the OOD classes  
        inputs[is_ood,:-1:2,start_ind:] = (e_fac*(self.mus_class_new[choices_c] + 
                self.eps*np.random.normal(size=(self.S,self.N, self.D))/np.sqrt(self.D)))[is_ood] 
        # Label item embeddings sampled from the OOD classes
        inputs[is_ood,1:-1:2,start_ind:] = ((self.mus_label[self.labels_class_new])[choices_c])[is_ood] 
        # Target item embedding sampled from the OOD classes
        inputs[is_ood,-1,start_ind:] = (e_fac*(self.mus_class_new[target_c] + 
                self.eps*np.random.normal(size=(self.S,self.D))/np.sqrt(self.D)))[is_ood] 
        
        # Vanilla Position Embeddings
        shifts = np.random.choice((2*self.Nmax + 1) - (2*self.N + 1) + 1, size=(self.S))
        
        labels = np.zeros((self.S,self.L),dtype= bool)
        target_classes = np.zeros(self.S, dtype = int)
        label_sequences = np.zeros((self.S, self.N+1), dtype=int)
        
        for s in range(self.S):
            if is_ood[s]:
                labels[s,self.labels_class_new[target_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(self.labels_class_new[choices_c[s]],self.labels_class_new[target_c[s]])
            else:
                labels[s,self.labels_class[target[s]]] = True
                target_classes[s] = target[s]
                label_sequences[s] = np.append(self.labels_class[choices[s]],self.labels_class[target[s]])
            if not self.rope:
                    inputs[s,:,shifts[s]:shifts[s] + 2*self.N+1] = np.identity(2*self.N+1)  
                
        if self.seq_labels:
            return torch.FloatTensor(inputs), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif self.output_target_labels:
            return torch.FloatTensor(inputs), torch.FloatTensor(labels), torch.FloatTensor(target_classes)
        else:
            return torch.FloatTensor(inputs), torch.FloatTensor(labels)
      
    def __iter__(self):
        while self.n_samples < self.datasize:
            inputs, labels = self.generate_sequence()
            # Convert to tensors and yield as a batch
            yield torch.FloatTensor(inputs), torch.FloatTensor(labels)
            self.n_samples += 1
