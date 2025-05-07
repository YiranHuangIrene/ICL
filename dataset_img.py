import numpy as np
import math
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.datasets import Omniglot
from dataclasses import dataclass, field
from scipy.ndimage import rotate

@dataclass
class OmniglotDatasetArgs:
    root: str = './'
    K: int = 256
    n_img_per_class: int = 15
    eps: float = 0.1
    alpha: float = 0
    augment: bool = False

    
class OmniglotDataset(IterableDataset):
    def __init__(self, args: OmniglotDatasetArgs, S: int, datasize: int):
        """
        IMAGE_SIZE = 105
        N_CHARACTER_CLASSES = 1623 (964 + 659)
        N_EXEMPLARS_PER_CLASS = 20
        """
        self.K = args.K
        self.n_img_per_class = args.n_img_per_class
        self.eps = args.eps
        self.alpha = args.alpha
        P = 1.0/(np.arange(1,self.K+1)**self.alpha)
        P /= np.sum(P)
        self.P = P
        self.augment = args.augment
        self.root = args.root
        self.S = S
        self.datasize = datasize
       
        
        from collections import defaultdict
        char_to_imgs = defaultdict(list)
        
        train_dataset = Omniglot(args.root, background=True)
        eval_dataset = Omniglot(args.root, background=False)
        
        for img, label in list(train_dataset):
            img_arr = np.array(img).astype(np.float32) / 255.0
            char_to_imgs[label].append(img_arr)
        len_train = len(char_to_imgs)
        for img, label in list(eval_dataset):
            label = label + len_train
            img_arr = np.array(img).astype(np.float32) / 255.0
            char_to_imgs[label].append(img_arr)
            
        all_labels = list(char_to_imgs.keys())
        selected_labels_train = all_labels[:self.K]
        self.class_to_imgs_train = {i: np.array(char_to_imgs[label])[:self.n_img_per_class] for i, label in enumerate(selected_labels_train)}
        self.class_to_imgs_eval = {i: np.array(char_to_imgs[label])[-5:] for i, label in enumerate(selected_labels_train)} # select last 5 images for evaluation      
        selected_labels_test = all_labels[-128:] # Hold out 128 classes for multimodal evaluation
        self.class_to_imgs_test = {i: np.array(char_to_imgs[label]) for i, label in enumerate(selected_labels_test)}
       

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            imgs, labels = self.generate_input()
            yield imgs, labels
            n += 1
            
            
    def generate_input(self):
        class_idxs = np.random.choice(self.K, size=(self.S,), p=self.P)
        batch_imgs = []
        for c in class_idxs:
            img_idx = np.random.randint(0, self.n_img_per_class)
            img = self.class_to_imgs_train[c][img_idx]
            batch_imgs.append(img)
        imgs = np.array(batch_imgs)
        # Add noise to images
        if self.eps > 0:
            noise = np.random.normal(0, self.eps, size=imgs.shape).astype(np.float32)
            imgs = imgs + noise
        # Apply augmentations by rotating the images by -10 to 10 degrees
        if self.augment:
            rotated = np.empty_like(imgs)
            for i, img in enumerate(imgs):
                angle = np.random.uniform(-10, 10)
                rotated[i] = rotate(img, angle, reshape=False, order=1, mode='reflect')
            imgs = rotated
        # Convert labels to one-hot encoding
        labels = np.eye(self.K)[class_idxs]
        return torch.FloatTensor(imgs), torch.FloatTensor(labels)
    
    def generate_val(self):
        imgs = []
        labels = []
        for c in range(self.K):
            imgs.append(self.class_to_imgs_eval[c])
            # Append self.class_to_imgs_eval[c].shape[0] times labels
            labels.append(c*np.ones(self.class_to_imgs_eval[c].shape[0],dtype=np.int32))
        imgs = np.stack(imgs, axis=0)
        imgs = imgs.reshape(imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3])
        labels = np.array(labels)
        # Convert labels to one-hot encoding
        labels = np.eye(self.K)[labels]
        labels = labels.reshape(-1, self.K)
        return torch.FloatTensor(imgs), torch.FloatTensor(labels)


class OmniglotMMDataset(IterableDataset):
    def __init__(self, imgargs, mus_label_m1, mus_class_m1, mus_label_m2, labels_class_m2, mapping_m2_to_m1, N,S, eps1= 0.1, eps2 = 0.1, B= 0, p_B = 0, P1 = None, P2 = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, seq_labels = False, datasize = 0):

        super().__init__()
        self.mus_label_m1 = mus_label_m1
        self.mus_class_m1 = mus_class_m1
        self.mus_label_m2 = mus_label_m2
        self.labels_class_m2 = labels_class_m2
        self.mapping_m2_to_m1 = mapping_m2_to_m1
        self.N = N
        self.S = S
        self.eps1 = eps1
        self.eps2 = eps2
        self.B = B
        self.p_B = p_B
        self.p_C = p_C
        self.flip_labels = flip_labels
        self.output_target_labels = output_target_labels
        self.no_repeats = no_repeats
        self.seq_labels = seq_labels
        self.datasize = datasize
        
        L1 = mus_label_m1.shape[0]
        L2 = mus_label_m2.shape[0]
        K1 = mus_class_m1.shape[0]
        D1 = mus_class_m1.shape[1]
        
        self.L1 = L1
        self.L2 = L2
        self.K1 = K1
        self.K2 = imgargs.K
        self.D1 = D1

            
        # Set up class distribution if not provided
        if P1 is None or len(P1) != K1:
            self.P1 = np.ones(K1)/K1
        else:
            self.P1 = P1    
        if P2 is None or len(P2) != self.K2:
            self.P2 = np.ones(self.K2)/self.K2
        else:
            self.P2 = P2
            
        # Set up OOD class embeddings
        self.K_c1 = 128
        if self.K_c1 < L1 or self.K_c1%L1 != 0:
            print("K_c1 > L1 and K_c1%L1 == 0 is required")
            raise ValueError("K_c1 > L1 and K_c1%L1 == 0 is required")

        # Validate burstiness parameter
        if B > 0 and N % B != 0:
            raise ValueError("N must be divisible by B when B > 0")
        if B >= N:
            raise ValueError("B must be less than N")
        if B == 0:
            self.B = int(N/2)
            self.p_B = 0
        
        # Set up image dataset
        self.n_img_per_class = imgargs.n_img_per_class
        self.eps0 = imgargs.eps
        self.alpha0 = imgargs.alpha
        self.augment = imgargs.augment
       
        from collections import defaultdict
        char_to_imgs = defaultdict(list)
        
        train_dataset = Omniglot(imgargs.root, background=True)
        eval_dataset = Omniglot(imgargs.root, background=False)
        
        for img, label in list(train_dataset):
            img_arr = np.array(img).astype(np.float32) / 255.0
            char_to_imgs[label].append(img_arr)
        len_train = len(char_to_imgs)
        for img, label in list(eval_dataset):
            label = label + len_train
            img_arr = np.array(img).astype(np.float32) / 255.0
            char_to_imgs[label].append(img_arr)
        img_size = char_to_imgs[0][0].shape[-1]
        n_img_all = len(char_to_imgs[0])
        self.n_img_all = n_img_all
        self.D2 = img_size
        
        all_labels = list(char_to_imgs.keys())
        selected_labels_train = all_labels[:self.K2]
        self.class_to_imgs_train = {i: np.array(char_to_imgs[label])[:self.n_img_per_class] for i, label in enumerate(selected_labels_train)}
        self.class_to_imgs_eval = {i: np.array(char_to_imgs[label])[-5:] for i, label in enumerate(selected_labels_train)} # select last 5 images for evaluation      
        selected_labels_test = all_labels[-128:] # Hold out 128 classes for multimodal evaluation
        self.class_to_imgs_test = {i: np.array(char_to_imgs[label]) for i, label in enumerate(selected_labels_test)}
        
    def generate_sequence(self):
        S = self.S
        N = self.N
        B = self.B
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = self.D2
        K_c1 = self.K_c1
        K_c2 = 128
        mus_class_m1 = self.mus_class_m1
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        p_B = self.p_B
        p_C = self.p_C
        eps1 = self.eps1
        eps2 = self.eps2
        flip_labels = self.flip_labels
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        n_per_class = self.n_img_per_class
        img_size = self.D2
        
        # Decide in final input dimension
        inputs_mm = np.zeros((S,3*N+1,D1))
        inputs_2 = np.zeros((S,N+1,D2,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        
        filt_C = np.random.uniform(size = S) > p_C
        
        def add_noise_to_img(imgs):
            noise = np.random.normal(0, eps2, imgs.shape)
            imgs = imgs + noise.astype(np.float32)
            return imgs
        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        inputs_mm[filt_C,:-1:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_train[c][idx]
        inputs_2[filt_C,:-1,:] = add_noise_to_img(sampled[filt_C])
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-1:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-1:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
     
        # Target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_train[c][idx] for c, idx in zip(targets, sample_idxs)], axis=0)
        inputs_2[filt_C,-1,:] = add_noise_to_img(sampled[filt_C])
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-1:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        sample_idxs = np.random.randint(0, self.n_img_all, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_c2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_test[c][idx]
        inputs_2[~filt_C,:-1] =  add_noise_to_img(sampled[~filt_C])
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-1:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, self.n_img_all, size=(S,))
        sampled = np.stack([self.class_to_imgs_test[c][idx] for c, idx in zip(targets_c, sample_idxs)], axis=0)
        inputs_2[~filt_C,-1] = add_noise_to_img(sampled[~filt_C])
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    def generate_test_sequence(self,S,B,p_B,p_C,flip_labels=False):
        N = self.N
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = self.D2
        K_c1 = self.K_c1
        K_c2 = 128
        mus_class_m1 = self.mus_class_m1
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        eps1 = self.eps1
        eps2 = self.eps2
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        n_per_class = self.n_img_per_class
        img_size = self.D2
        
        # Decide in final input dimension
        inputs_mm = np.zeros((S,3*N+1,D1))
        inputs_2 = np.zeros((S,N+1,D2,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        
        filt_C = np.random.uniform(size = S) > p_C
        
        def add_noise_to_img(imgs):
            noise = np.random.normal(0, eps2, imgs.shape)
            imgs = imgs + noise.astype(np.float32)
            return imgs
        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        inputs_mm[filt_C,:-1:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_train[c][idx]
        inputs_2[filt_C,:-1,:] = add_noise_to_img(sampled[filt_C])
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-1:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-1:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
     
        # Target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_train[c][idx] for c, idx in zip(targets, sample_idxs)], axis=0)
        inputs_2[filt_C,-1,:] = add_noise_to_img(sampled[filt_C])
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-1:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        sample_idxs = np.random.randint(0, self.n_img_all, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_c2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_test[c][idx]
        inputs_2[~filt_C,:-1] =  add_noise_to_img(sampled[~filt_C])
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-1:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, self.n_img_all, size=(S,))
        sampled = np.stack([self.class_to_imgs_test[c][idx] for c, idx in zip(targets_c, sample_idxs)], axis=0)
        inputs_2[~filt_C,-1] = add_noise_to_img(sampled[~filt_C])
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            inputs_m, inputs_2, labels = self.generate_sequence()
            yield inputs_m, inputs_2, labels
            n += 1
    
    
