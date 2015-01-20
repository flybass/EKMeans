import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta

class EKMeans(object):
    #constructor - has user choice parameters
    def __init__(self, epsilon = .001, sharpness = .1, tol = .0001, max_iter=50, noise_tol =4, one_clust_test=True):
        self.epsilon = epsilon
        self.sharpness = sharpness
        self.tolerance = tol
        self.max_iter = max_iter
        self.noise_tol = noise_tol
        self.one_clust_test = one_clust_test
        #attributes - user does not set
        self.reset_obj()
        
    def reset_obj(self):
        self.cluster_centers_ = []
        self.labels_ = None
        self.n_centers_ = 0
        self.noise_row_drops = []
        self.inertia_ =0 
        self.gap_stat_ = 0
        
    #fit routine
    def fit(self,A):
        self.reset_obj()
        #1.  Create the Matrix L = D ^ -.5 * A * D^-.5; D = diagonal matrix that is sum of rows in A
        D = np.diag(np.sum(A, axis=1))
        D_power = D**-.5
        D_power[D_power == np.inf] = 0
        L = np.dot(D_power,np.dot(A,D_power))
        #1. Compute the EigenVectors and Values of L
        #print L.shape
        e_vals, e_vecs  = np.linalg.eigh(L)
        #2. Get Reverse Sorted Order  - largest to smallest
        e_order = np.argsort(e_vals)[::-1]
        one_c_fit = 10**10
        if len(e_vals) ==1:
            self.labels_ = np.zeros(1)
            return self.labels_
        if self.one_clust_test == True:
            self.one_clust_fit_alt(e_vecs, e_order)
            one_c_fit = self.gap_stat_
        flag =1
        last_loop=0
        while flag:
            origin_pts = [[1]]
            centers = []
            #run fit routine as long as there are points clustered at O
            while origin_pts and len(centers)<len(e_order):
                eig_vecs_matrix, centers = self.initialize_fit(e_vecs, e_order, centers)
                centers, classified = self.k_iter(centers, eig_vecs_matrix)
                #print eig_vecs_matrix
                if len(centers) -1 in classified:
                    origin_pts = classified[len(centers)-1]
                else:
                    origin_pts = []
            tol_runs =0
            if last_loop !=1:
                contd = 0
                while tol_runs < self.noise_tol and len(centers) <= len(e_order):
                    if contd ==1:
                        tol_runs +=1
                        continue
                    eig_vecs_matrix, centers = self.initialize_fit(e_vecs, e_order, centers)
                    #eig_vecs_matrix = np.delete(eig_vecs_matrix, self.noise_row_drops, 0)
                    centers, classified = self.k_iter(centers, eig_vecs_matrix)
                    #print classified
                    if len(centers) -1 in classified:
                        bad_vecs = classified[len(centers)-1]
                        bad_inds = self.ret_bad_inds(bad_vecs, e_vecs,e_order, centers)
                        self.noise_row_drops.extend(bad_inds)
                        contd=1
                        continue   
                    tol_runs +=1
                last_loop =1
            else:
                self.final_fit(centers,e_vecs, e_order)
                flag = 0
        if one_c_fit < self.gap_stat_:
            last_mult = self.gap_stat_
            last_one = one_c_fit
            ref_mult, ref_sing = self.reference_sim( A, classified,self.labels_)
            adj_mult = ref_mult - last_mult
            adj_one = ref_sing - last_one
            if adj_one< adj_mult:
                self.one_clust_fit_alt(e_vecs, e_order)
            else:
                self.one_clust_test = False
                self.fit(A)
            self.one_clust_test = True
        return self.labels_
    
    def ret_bad_inds(self,bad_vecs, e_vecs, e_order, centers):
        bad_inds = []
        eig_vecs_matrix = np.transpose(np.array([e_vecs[:,e_order[o]] for o in range(0,len(centers)-1) ]))
        for n in range(0,eig_vecs_matrix.shape[0]):
            v = eig_vecs_matrix[n,:]
            if any((v == a).all() for a in bad_vecs):
                bad_inds.append(n)
        return bad_inds
            
    def final_fit(self,centers,e_vecs, e_order):
        #print centers
        centers = centers[:len(centers)-1]
        #print centers
        q =len(centers)
        #print q
        center_classes = [np.dot(c,c) > self.epsilon for c in centers]
        m_s = [self.e_dist_constants(c) for c in centers]
        eig_vecs_matrix = np.transpose(np.array([e_vecs[:,e_order[o]] for o in range(0,q) ]))
        classified = self.classify_pts(eig_vecs_matrix, centers, center_classes, m_s)
        #print classified
        labels = self.set_labels(classified,eig_vecs_matrix)
        #print classified
        self.inertia_,self.gap_stat_ = self.fit_quality(classified,centers,labels)
        self.labels_ = np.array(labels)
        self.n_centers_ = len(set(labels))
        self.cluster_centers_ =centers
        return  
    
    def set_labels(self,classified, eig_vecs_matrix):
        labels = []
        for i in range(0,eig_vecs_matrix.shape[0]):
            x = eig_vecs_matrix[i,:]
            for c in classified.keys():
                listy = classified[c]
                if any((x == a).all() for a in listy):
                    labels.append(c)
        return labels
        
    def k_iter(self,centers, eig_vecs_matrix ):
        mov = 1
        classified = dict()
        iterations = 0
        while (mov > self.tolerance) and (iterations < self.max_iter):
            #1.  classify the centers as near or far away from origin
            center_classes = [np.dot(c,c) > self.epsilon for c in centers]
            m_s = [self.e_dist_constants(c) for c in centers]
            #2.  classify the center belonging to each x
            #looks like classified[center index in list] = list of x's
            classified = self.classify_pts(eig_vecs_matrix, centers, center_classes, m_s)
            new_centers = self.new_center_calc(classified,centers)
            #4.  Calculate maximum movement
            mov = max([self.or_dist(centers[i],new_centers[i]) for i in range(0,len(centers))])
            centers = new_centers
            iterations +=1
        return centers,classified
    
    def classify_pts(self, eig_vecs_matrix, centers, center_classes, m_s):
        classified = dict()
        for i in range(0,eig_vecs_matrix.shape[0]):
            x = eig_vecs_matrix[i,:]
            #print eig_vecs_matrix.shape[0]
            closest_center = self.closest(x,centers,center_classes, m_s)
            if closest_center in classified:   
                classified[closest_center].append(x)
            else:
                classified[closest_center] = [x]
        return classified
        
    def new_center_calc(self,classified, centers):
        new_centers = []
        for k in range(0,len(centers)):
            if k in classified:
                new_centers.append(np.mean(np.array(classified[k]),axis=0))
            else:
                #unsure about this - not changing center vs setting center to origin
                new_centers.append(centers[k])
        return new_centers
    
    def closest(self,x,centers, center_classes, m_s):
        closest_dist = 10**9
        chosen = 0
        for ind in range(0,len(centers)):
            center = centers[ind]
            #print "Center is ", center
            center_class = center_classes[ind]
            if center_class ==1:
                dist = self.e_dist(x,center, m_s[ind])
                #print "E-dist is", dist
            else:
                dist = self.or_dist(x,center)
                #print "OR-dist is", dist
            if dist < closest_dist:
                chosen = ind
                closest_dist = dist
        #print "Chosen Ind is", chosen
        return chosen
           
    def e_dist(self,x,c, M):
        x_minus_c = np.array([np.array(x) - np.array(c)])
        return np.dot(x_minus_c,np.dot(M,x_minus_c.T))

    def e_dist_constants(self,c):
        cT_c = np.dot(np.array(c),np.array(c))
        c_cT = np.dot(np.array([np.array(c)]).T, np.array([np.array(c)]))
        M = (1/self.sharpness) * (np.identity(len(c)) - (1/cT_c)* c_cT ) + (self.sharpness/cT_c)* c_cT
        return M

    def or_dist(self,x,c):
        return np.linalg.norm(np.array(x)-np.array(c))
    
    def initialize_fit(self,e_vecs, e_order,centers):
        if not centers:
            curr_centroids = 2
            #print e_vecs
            eig_vecs_matrix = np.transpose(np.array([e_vecs[:,e_order[o]] for o in range(0,curr_centroids) ]))
            eig_vecs_matrix = np.delete(eig_vecs_matrix, self.noise_row_drops, 0)
            #Last center is always initialized at the origin
            c3 = np.zeros(curr_centroids)
            #c1 is initialized at the piont that is furthest away from the origin
            c1 = np.zeros(curr_centroids)
            for n in range(0,eig_vecs_matrix.shape[0]):
                v = eig_vecs_matrix[n,:]
                if np.linalg.norm(v) > np.linalg.norm(c1):
                    c1 = v
            #c2 is initialized at the point that simultaneously maximizes its norm while
            #minimizing the dot product with c1
            c2 = np.zeros(curr_centroids)
            for n in range(0,eig_vecs_matrix.shape[0]):
                small = .000000000001
                v = eig_vecs_matrix[n,:]
                if np.linalg.norm(v) / np.sqrt(max(np.dot(c1,v),small)) > np.linalg.norm(c2)/ np.sqrt(max(np.dot(c1,c2),small)):
                    c2 = v
            centers.extend([c1,c2,c3])
        else:
            centers  = [np.append(c, 0) for c in centers]
            curr_centroids = len(centers)
            eig_vecs_matrix = np.transpose(np.array([e_vecs[:,e_order[o]] for o in range(0,curr_centroids) ]))
            eig_vecs_matrix = np.delete(eig_vecs_matrix, self.noise_row_drops, 0)
            centers.append(np.zeros(curr_centroids))
        return eig_vecs_matrix, centers
    
    def e_dist_alt(self,x,c):
        cT_c = np.dot(np.array(c),np.array(c))
        c_cT = np.dot(np.array([np.array(c)]).T, np.array([np.array(c)]))
        M = (1/self.sharpness) * (np.identity(len(c)) - (1/cT_c)* c_cT ) + (self.sharpness/cT_c)* c_cT
        x_minus_c = np.array([np.array(x) - np.array(c)])
        return np.dot(x_minus_c,np.dot(M,x_minus_c.T))

    def fit_quality(self,classified,centers,labels):
        center_classes = [np.dot(c,c) > self.epsilon for c in centers]
        m_s = [self.e_dist_constants(c) for c in centers]
        inertia=0
        W_k = 0
        for i in classified.keys():
            D_r = 0
            if i in classified:
                x_s = classified[i]
            else:
                x_s = []
            n_r = len(x_s)
            for k in range(0,len(x_s)):
                x1 = x_s[k]
                if center_classes[i] ==1:
                    inertia += self.e_dist(x1,centers[i], m_s[i])
                else:
                    inertia += self.or_dist(x1,centers[i])
                for j in range(0,k):
                    x2 = x_s[j]
                    if center_classes[i] ==1:
                        D_r+= self.e_dist_alt(x1,x2)**2
                        D_r+= self.e_dist_alt(x2,x1)**2
                    else:
                        D_r+= self.or_dist(x1,x2)**2
                        D_r+= self.or_dist(x2,x1)**2
            D_r = float(D_r)/(2*n_r)
            W_k += D_r
        #print classified
        #print classified #classifiedprint classified
        return inertia, W_k
    
    def one_clust_fit_alt(self,e_vecs, e_order):
        eig_vecs_matrix = np.transpose(np.array([e_vecs[:,e_order[o]] for o in range(0,1) ]))
        #eig_vecs_matrix = np.delete(eig_vecs_matrix, self.noise_row_drops, 0)
        c2 = np.zeros(1)
        c1 = np.zeros(1)
        for n in range(0,eig_vecs_matrix.shape[0]):
            v = eig_vecs_matrix[n,:]
            if np.linalg.norm(v) > np.linalg.norm(c1):
                c1 = v
        centers=[c1,c2]
        centers,classified =  self.k_iter(centers, eig_vecs_matrix )
        if len(centers) -1 in classified:
            return False
        else:
            self.labels_ = np.zeros(eig_vecs_matrix.shape[0])
            self.inertia_,self.gap_stat_=self.fit_quality(classified,centers,self.labels_)
            self.n_centers_ = 1
            return True

    def reference_sim(self, A, classified,labels):
        num_centers = len(set(labels))
        small = .0000000000001
        ideal_A = np.zeros([A.shape[0],A.shape[1]])
        for i in range(0,len(labels)):
            for j in range(0,i+1):
                if labels[i] == labels[j]:
                    ideal_A[i,j] = 1
                    ideal_A[j,i] = 1         
        pred_pos = A[ideal_A ==1]
        pred_neg = A[ideal_A ==0]
        pos_a,pos_b,pos_loc, pos_scale= beta.fit(pred_pos)
        neg_a,neg_b,neg_loc, neg_scale= beta.fit(pred_neg)
        fits = []
        #Fit comparison iwth more than 1 clust
        for sim in range(0, 50):
            simulated_mat = np.ones([A.shape[0],A.shape[1]])
            for i in range(0,len(labels)):
                for j in range(0,i):
                    if ideal_A[i,j] ==0:
                        simulated_mat[i,j] = simulated_mat[j,i]= beta.rvs(max(neg_a,small), max(small,neg_b), loc=neg_loc,scale =neg_scale)
                    else:
                        simulated_mat[i, j ] = simulated_mat[j, i ] = beta.rvs(max(pos_a,small), max(small,pos_b), loc=pos_loc,scale =pos_scale)
            self.one_clust_test = False
            whereAreNaNs = np.isnan(simulated_mat)
            simulated_mat[whereAreNaNs] = 0
            self.fit(simulated_mat)
            #print simulated_mat
            fits.append(self.gap_stat_)
        multi_fit = np.mean(fits)
        fits_one = []
        pos_a,pos_b,pos_loc, pos_scale= beta.fit(A)
        for sim in range(0, 50):
            simulated_mat = np.ones([A.shape[0],A.shape[1]])
            for i in range(0,len(labels)):
                for j in range(0,i):
                    simulated_mat[i,j] = simulated_mat[j,i]= beta.rvs(max(small,pos_a), max(small,pos_b), loc=pos_loc,scale =pos_scale)
            whereAreNaNs = np.isnan(simulated_mat)
            simulated_mat[whereAreNaNs] = 0
            e_vals, e_vecs  = np.linalg.eigh(simulated_mat)
            #2. Get Reverse Sorted Order  - largest to smallest
            e_order = np.argsort(e_vals)[::-1]
            self.one_clust_fit_alt(e_vecs,e_order)
            fits_one.append(self.gap_stat_)
        one_fit = np.mean(fits_one)
        return multi_fit, one_fit
