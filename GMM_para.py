#this generate GMMs from w,y,z and manifold parameters, also including sampling
import tensorflow as tf
import numpy as np
from scipy.special import softmax

class GaussianMixture_LS():

    def __init__(self,w,y,z,b,beta,C,mu,xi,n_gmm,dim,K_r,K_b):
        #calculate weight
        #self.weight=tf.nn.softmax(tf.sigmoid(tf.matmul(w,xi)),axis=1)
        self.weight = tf.nn.softmax(tf.matmul(w, xi), axis=1)
        #self.weight=tf.Print(weight,[weight])
        self.weight = tf.identity(self.weight, name="weight_hat")
        #shape n_gmm*K_r
        #calculate mu
        self.mean=tf.tensordot(z,mu,axes=[[1],[0]])+tf.tile(tf.expand_dims(b,axis=0),[n_gmm,1,1])
        self.mean = tf.identity(self.mean, name="mean_hat")
        #shape n_gmm*dim*K_r

        #calculate Sigma log(1+exp(y)): shape n_gmm*dim*dim*K_r
        I_mat=tf.tile(tf.expand_dims(tf.eye(dim,batch_shape=[n_gmm]),axis=3),[1,1,1,K_r])
        C_squ=tf.linalg.matmul(tf.transpose(C,perm=[3,0,1,2]),tf.transpose(C,perm=[3,0,2,1]))
        self.Prcs=tf.tensordot(tf.sigmoid(-y),tf.transpose(C_squ,perm=[1,2,3,0]),axes=[[1],[0]])+tf.multiply(tf.tile(tf.square(beta),[n_gmm,dim,dim,1]),I_mat)
        self.Prcs = tf.identity(self.Prcs, name="Prcs_hat")

        #calculate log det
        self.logdet_q=tf.tile(tf.expand_dims(tf.linalg.logdet(tf.transpose(self.Prcs,perm=[0,3,1,2])),axis=1),[1,K_b,1])


class GaussianMixture_data():

    def __init__(self, mean, Sigma, weight,KL_m):
        #n_gmm*K_b
        self.weight=tf.transpose(tf.constant(weight))
        #self.unif_weight=tf.ones([n_gmm,K_r],tf.float32)/tf.constant(K_r,dtype=tf.float32)
        #print(self.weight)
        #n_gmm*dim*dim*K_b
        self.Sigma=tf.transpose(tf.constant(Sigma),[3,0,1,2])
        #print(self.Prcs)
        #n_gmm*dim*K_b
        self.mean=tf.transpose(tf.constant(mean),[2,0,1])
        #print(self.mean)

        self.KL_m=tf.transpose(tf.constant(KL_m))

        #self.logdet_p=tf.expand_dims(tf.linalg.logdet(tf.transpose(self.Prcs,perm=[0,3,1,2])),axis=2)



class GaussianMixture_HLS():

    def __init__(self,v,H_w,H_y,H_z,b,beta,C,mu,xi,n_gmm,dim,K_r,K_b):
        #calculate weight
        #self.weight=tf.nn.softmax(tf.sigmoid(tf.matmul(w,xi)),axis=1)

        self.w=tf.matmul(v,H_w)
        self.y=tf.matmul(v,H_y)
        self.z=tf.matmul(v,H_z)

        self.weight = tf.nn.softmax(tf.matmul(self.w, xi), axis=1)
        #self.weight=tf.Print(weight,[weight])
        self.weight = tf.identity(self.weight, name="weight_hat")
        #shape n_gmm*K_r
        #calculate mu
        self.mean=tf.tensordot(self.z,mu,axes=[[1],[0]])+tf.tile(tf.expand_dims(b,axis=0),[n_gmm,1,1])
        self.mean = tf.identity(self.mean, name="mean_hat")
        #shape n_gmm*dim*K_r

        #calculate Sigma log(1+exp(y)): shape n_gmm*dim*dim*K_r
        I_mat=tf.tile(tf.expand_dims(tf.eye(dim,batch_shape=[n_gmm]),axis=3),[1,1,1,K_r])
        C_squ=tf.linalg.matmul(tf.transpose(C,perm=[3,0,1,2]),tf.transpose(C,perm=[3,0,2,1]))
        self.Prcs=tf.tensordot(tf.sigmoid(-self.y),tf.transpose(C_squ,perm=[1,2,3,0]),axes=[[1],[0]])+tf.multiply(tf.tile(tf.square(beta),[n_gmm,dim,dim,1]),I_mat)
        self.Prcs = tf.identity(self.Prcs, name="Prcs_hat")

        # calculate log det
        # n_gmm*K_b*K_r
        self.logdet_q=tf.tile(tf.expand_dims(tf.linalg.logdet(tf.transpose(self.Prcs,perm=[0,3,1,2])),axis=1),[1,K_b,1])


class GaussianMixture_KHLS():

    def __init__(self,v,ps_v, n_ps_v, lamda_kern, gamma_width, gamma_scale, poly_scale, H_w, H_y, H_z, b, beta, C, mu, xi, n_gmm, dim, K_r, K_b):
        #calculate weight
        #self.weight=tf.nn.softmax(tf.sigmoid(tf.matmul(w,xi)),axis=1)

        # 1. calculate the exponential kernel matrix for pseudo-points
        # diff_v: n_ps_v*n_ps_v
        square_diff_ps_v=tf.reduce_sum(tf.square(tf.expand_dims(ps_v,axis=1)-tf.expand_dims(ps_v,axis=0)),axis=2)
        exp_kernel_ps_v=gamma_scale*gamma_scale*tf.exp(-square_diff_ps_v/(2.0*gamma_width*gamma_width))

        # 2. calculate the polynomial kernel matrix for pseudo-points
        poly_kernel_ps_v=poly_scale*poly_scale*tf.reduce_sum(tf.multiply(tf.expand_dims(ps_v,axis=1),tf.expand_dims(ps_v,axis=0)),axis=2)

        Kern_ps_v=exp_kernel_ps_v+poly_kernel_ps_v+lamda_kern*tf.eye(n_ps_v)

        # 3. calculate the exponential kernel matrix for pseudo-points and latent variables
        # n_ps_v*n_gmm
        square_diff_v=tf.reduce_sum(tf.square(tf.expand_dims(ps_v,axis=1)-tf.expand_dims(v,axis=0)),axis=2)
        exp_kernel_v=gamma_scale*gamma_scale*tf.exp(-square_diff_v/(2.0*gamma_width*gamma_width))


        # 4. calculate the polynomial kernel matrix for pseudo-points and latent variables
        # n_ps_v*n_gmm
        poly_kernel_v=poly_scale*poly_scale*tf.reduce_sum(tf.multiply(tf.expand_dims(ps_v,axis=1),tf.expand_dims(v,axis=0)),axis=2)

        Kern_v=exp_kernel_v+poly_kernel_v

        # 5. calculate w, y and z

        Kern_HLS=tf.matmul(tf.linalg.inv(Kern_ps_v),Kern_v)

        self.w=tf.transpose(tf.matmul(H_w,Kern_HLS))
        self.y=tf.transpose(tf.matmul(H_y,Kern_HLS))
        self.z=tf.transpose(tf.matmul(H_z,Kern_HLS))

        self.weight = tf.nn.softmax(tf.matmul(self.w, xi), axis=1)
        #self.weight=tf.Print(weight,[weight])
        self.weight = tf.identity(self.weight, name="weight_hat")
        #shape n_gmm*K_r
        #calculate mu
        self.mean=tf.tensordot(self.z,mu,axes=[[1],[0]])+tf.tile(tf.expand_dims(b,axis=0),[n_gmm,1,1])
        self.mean = tf.identity(self.mean, name="mean_hat")
        #shape n_gmm*dim*K_r

        #calculate Sigma log(1+exp(y)): shape n_gmm*dim*dim*K_r
        I_mat=tf.tile(tf.expand_dims(tf.eye(dim,batch_shape=[n_gmm]),axis=3),[1,1,1,K_r])
        C_squ=tf.linalg.matmul(tf.transpose(C,perm=[3,0,1,2]),tf.transpose(C,perm=[3,0,2,1]))
        self.Prcs=tf.tensordot(tf.sigmoid(-self.y),tf.transpose(C_squ,perm=[1,2,3,0]),axes=[[1],[0]])+tf.multiply(tf.tile(tf.square(beta),[n_gmm,dim,dim,1]),I_mat)
        self.Prcs = tf.identity(self.Prcs, name="Prcs_hat")

        # calculate log det
        # n_gmm*K_b*K_r
        self.logdet_q=tf.tile(tf.expand_dims(tf.linalg.logdet(tf.transpose(self.Prcs,perm=[0,3,1,2])),axis=1),[1,K_b,1])


class GaussianMixture_KHLS_NN():

    def __init__(self,v,ps_v, n_ps_v, lamda_kern, gamma_width, gamma_scale, poly_scale, H_w, H_y, H_z,beta, C, mu_1,b_1,mu_out,b_out, xi, n_gmm, dim, K_r, K_b):
        #calculate weight
        #self.weight=tf.nn.softmax(tf.sigmoid(tf.matmul(w,xi)),axis=1)

        # 1. calculate the exponential kernel matrix for pseudo-points
        # diff_v: n_ps_v*n_ps_v
        square_diff_ps_v=tf.reduce_sum(tf.square(tf.expand_dims(ps_v,axis=1)-tf.expand_dims(ps_v,axis=0)),axis=2)
        exp_kernel_ps_v=gamma_scale*gamma_scale*tf.exp(-square_diff_ps_v/(2.0*gamma_width*gamma_width))

        # 2. calculate the polynomial kernel matrix for pseudo-points
        poly_kernel_ps_v=poly_scale*poly_scale*tf.reduce_sum(tf.multiply(tf.expand_dims(ps_v,axis=1),tf.expand_dims(ps_v,axis=0)),axis=2)

        Kern_ps_v=exp_kernel_ps_v+poly_kernel_ps_v+lamda_kern*tf.eye(n_ps_v)

        # 3. calculate the exponential kernel matrix for pseudo-points and latent variables
        # n_ps_v*n_gmm
        square_diff_v=tf.reduce_sum(tf.square(tf.expand_dims(ps_v,axis=1)-tf.expand_dims(v,axis=0)),axis=2)
        exp_kernel_v=gamma_scale*gamma_scale*tf.exp(-square_diff_v/(2.0*gamma_width*gamma_width))


        # 4. calculate the polynomial kernel matrix for pseudo-points and latent variables
        # n_ps_v*n_gmm
        poly_kernel_v=poly_scale*poly_scale*tf.reduce_sum(tf.multiply(tf.expand_dims(ps_v,axis=1),tf.expand_dims(v,axis=0)),axis=2)

        Kern_v=exp_kernel_v+poly_kernel_v

        # 5. calculate w, y and z

        Kern_HLS=tf.matmul(tf.linalg.inv(Kern_ps_v),Kern_v)

        self.w=tf.transpose(tf.matmul(H_w,Kern_HLS))
        self.y=tf.transpose(tf.matmul(H_y,Kern_HLS))
        self.z=tf.transpose(tf.matmul(H_z,Kern_HLS))

        self.weight = tf.nn.softmax(tf.matmul(self.w, xi), axis=1)
        #self.weight=tf.Print(weight,[weight])
        self.weight = tf.identity(self.weight, name="weight_hat")
        #shape n_gmm*K_r


        #calculate mu
        #shape n_gmm*d_mu_1*K_r
        mean_hidden=tf.nn.sigmoid(tf.tensordot(self.z,mu_1,axes=[[1],[0]])+tf.tile(tf.expand_dims(b_1,axis=0),[n_gmm,1,1]))
        #shape n_gmm*d_mu_1*dim*K_r
        mean_out_prod=tf.multiply(tf.tile(tf.expand_dims(mean_hidden,axis=2),[1,1,dim,1]),tf.tile(tf.expand_dims(mu_out,axis=0),[n_gmm,1,1,1]))
        self.mean=tf.reduce_sum(mean_out_prod,axis=1)+tf.tile(b_out,[n_gmm,1,1])
        #self.mean=tf.tensordot(self.z,mu,axes=[[1],[0]])+tf.tile(tf.expand_dims(b,axis=0),[n_gmm,1,1])
        self.mean = tf.identity(self.mean, name="mean_hat")
        #shape n_gmm*dim*K_r

        #calculate Sigma log(1+exp(y)): shape n_gmm*dim*dim*K_r
        I_mat=tf.tile(tf.expand_dims(tf.eye(dim,batch_shape=[n_gmm]),axis=3),[1,1,1,K_r])
        C_squ=tf.linalg.matmul(tf.transpose(C,perm=[3,0,1,2]),tf.transpose(C,perm=[3,0,2,1]))
        self.Prcs=tf.tensordot(tf.log(1+tf.exp(self.y)),tf.transpose(C_squ,perm=[1,2,3,0]),axes=[[1],[0]])+tf.multiply(tf.tile(tf.square(beta),[n_gmm,dim,dim,1]),I_mat)
        self.Prcs = tf.identity(self.Prcs, name="Prcs_hat")

        # calculate log det
        # n_gmm*K_b*K_r
        self.logdet_q=tf.tile(tf.expand_dims(tf.linalg.logdet(tf.transpose(self.Prcs,perm=[0,3,1,2])),axis=1),[1,K_b,1])


class GaussianMixture_KHLS_diag_cov():

    def __init__(self, v, ps_v, n_ps_v, lamda_kern, gamma_width, gamma_scale, poly_scale, H_w, H_y, H_z, b, C_1, C_b_1, C_out, C_out_b, mu, xi, n_gmm, dim, K_r, K_b):
        #calculate weight
        #self.weight=tf.nn.softmax(tf.sigmoid(tf.matmul(w,xi)),axis=1)

        # 1. calculate the exponential kernel matrix for pseudo-points
        # diff_v: n_ps_v*n_ps_v
        square_diff_ps_v=tf.reduce_sum(tf.square(tf.expand_dims(ps_v,axis=1)-tf.expand_dims(ps_v,axis=0)),axis=2)
        exp_kernel_ps_v=gamma_scale*gamma_scale*tf.exp(-square_diff_ps_v/(2.0*gamma_width*gamma_width))

        # 2. calculate the polynomial kernel matrix for pseudo-points
        poly_kernel_ps_v=poly_scale*poly_scale*tf.reduce_sum(tf.multiply(tf.expand_dims(ps_v,axis=1),tf.expand_dims(ps_v,axis=0)),axis=2)

        Kern_ps_v=exp_kernel_ps_v+poly_kernel_ps_v+lamda_kern*tf.eye(n_ps_v)

        # 3. calculate the exponential kernel matrix for pseudo-points and latent variables
        # n_ps_v*n_gmm
        square_diff_v=tf.reduce_sum(tf.square(tf.expand_dims(ps_v,axis=1)-tf.expand_dims(v,axis=0)),axis=2)
        exp_kernel_v=gamma_scale*gamma_scale*tf.exp(-square_diff_v/(2.0*gamma_width*gamma_width))


        # 4. calculate the polynomial kernel matrix for pseudo-points and latent variables
        # n_ps_v*n_gmm
        poly_kernel_v=poly_scale*poly_scale*tf.reduce_sum(tf.multiply(tf.expand_dims(ps_v,axis=1),tf.expand_dims(v,axis=0)),axis=2)

        Kern_v=exp_kernel_v+poly_kernel_v

        # 5. calculate w, y and z

        Kern_HLS=tf.matmul(tf.linalg.inv(Kern_ps_v),Kern_v)

        self.w=tf.transpose(tf.matmul(H_w,Kern_HLS))
        self.y=tf.transpose(tf.matmul(H_y,Kern_HLS))
        self.z=tf.transpose(tf.matmul(H_z,Kern_HLS))


        self.weight = tf.nn.softmax(tf.matmul(self.w, xi), axis=1)
        #self.weight=tf.Print(weight,[weight])
        self.weight = tf.identity(self.weight, name="weight_hat")
        #shape n_gmm*K_r
        #calculate mu
        self.mean=tf.tensordot(self.z,mu,axes=[[1],[0]])+tf.tile(tf.expand_dims(b,axis=0),[n_gmm,1,1])
        self.mean = tf.identity(self.mean, name="mean_hat")
        #shape n_gmm*dim*K_r

        #calculate Precision matrix, shape n_gmm*dim*K_r
        # Prcs_lay_1: n_gmm*d_C_1*K_r
        Prcs_lay_1=tf.nn.relu(tf.tensordot(self.y,C_1,axes=[1,0])+tf.tile(C_b_1,[n_gmm,1,1]))

        #Prcs_lay_2=tf.nn.relu(tf.tensordot(Prcs_lay_1,C_2,axes=[1,0])+tf.tile(C_b_2,[n_gmm,1,1]))
        Prcs_prod=tf.multiply(tf.tile(tf.expand_dims(Prcs_lay_1,axis=2),[1,1,dim,1]),tf.tile(tf.expand_dims(C_out,axis=0),[n_gmm,1,1,1]))
        self.Prcs=tf.exp(tf.reduce_sum(Prcs_prod,axis=1)+tf.tile(C_out_b,[n_gmm,1,1]))

        #I_mat=tf.tile(tf.expand_dims(tf.eye(dim,batch_shape=[n_gmm]),axis=3),[1,1,1,K_r])
        #C_squ=tf.linalg.matmul(tf.transpose(C,perm=[3,0,1,2]),tf.transpose(C,perm=[3,0,2,1]))
        #self.Prcs=tf.tensordot(tf.sigmoid(-y),tf.transpose(C_squ,perm=[1,2,3,0]),axes=[[1],[0]])+tf.multiply(tf.tile(tf.square(beta),[n_gmm,dim,dim,1]),I_mat)
        #self.Prcs = tf.identity(self.Prcs, name="Prcs_hat")
        #calculate log det
        #with tf.device('/cpu:0'):
        logdet_q_reduce=tf.reduce_sum(tf.log(tf.transpose(self.Prcs,perm=[0,2,1])),axis=2)
        self.logdet_q=tf.tile(tf.expand_dims(logdet_q_reduce,axis=1),[1,K_b,1])


class GaussianMixture_HLS_diag_cov():

    def __init__(self,v,H_w,H_y,H_z,b,C_1,C_b_1,C_out,C_out_b,mu,xi,n_gmm,dim,K_r,K_b):
        #calculate weight
        #self.weight=tf.nn.softmax(tf.sigmoid(tf.matmul(w,xi)),axis=1)

        self.w=tf.matmul(v,H_w)
        self.y=tf.matmul(v,H_y)
        self.z=tf.matmul(v,H_z)

        self.weight = tf.nn.softmax(tf.matmul(self.w, xi), axis=1)
        #self.weight=tf.Print(weight,[weight])
        self.weight = tf.identity(self.weight, name="weight_hat")
        #shape n_gmm*K_r
        #calculate mu
        self.mean=tf.tensordot(self.z,mu,axes=[[1],[0]])+tf.tile(tf.expand_dims(b,axis=0),[n_gmm,1,1])
        self.mean = tf.identity(self.mean, name="mean_hat")
        #shape n_gmm*dim*K_r

        #calculate Precision matrix, shape n_gmm*dim*K_r
        # Prcs_lay_1: n_gmm*d_C_1*K_r
        Prcs_lay_1=tf.nn.relu(tf.tensordot(self.y,C_1,axes=[1,0])+tf.tile(C_b_1,[n_gmm,1,1]))

        #Prcs_lay_2=tf.nn.relu(tf.tensordot(Prcs_lay_1,C_2,axes=[1,0])+tf.tile(C_b_2,[n_gmm,1,1]))
        Prcs_prod=tf.multiply(tf.tile(tf.expand_dims(Prcs_lay_1,axis=2),[1,1,dim,1]),tf.tile(tf.expand_dims(C_out,axis=0),[n_gmm,1,1,1]))
        self.Prcs=tf.exp(tf.reduce_sum(Prcs_prod,axis=1)+tf.tile(C_out_b,[n_gmm,1,1]))

        #I_mat=tf.tile(tf.expand_dims(tf.eye(dim,batch_shape=[n_gmm]),axis=3),[1,1,1,K_r])
        #C_squ=tf.linalg.matmul(tf.transpose(C,perm=[3,0,1,2]),tf.transpose(C,perm=[3,0,2,1]))
        #self.Prcs=tf.tensordot(tf.sigmoid(-y),tf.transpose(C_squ,perm=[1,2,3,0]),axes=[[1],[0]])+tf.multiply(tf.tile(tf.square(beta),[n_gmm,dim,dim,1]),I_mat)
        #self.Prcs = tf.identity(self.Prcs, name="Prcs_hat")
        #calculate log det
        #with tf.device('/cpu:0'):
        logdet_q_reduce=tf.reduce_sum(tf.log(tf.transpose(self.Prcs,perm=[0,2,1])),axis=2)
        self.logdet_q=tf.tile(tf.expand_dims(logdet_q_reduce,axis=1),[1,K_b,1])

class GaussianMixture_data_diag_cov():

    def __init__(self, mean, Sigma, weight,KL_m):
        #n_gmm*K_b
        self.weight=tf.transpose(tf.constant(weight))
        #self.unif_weight=tf.ones([n_gmm,K_r],tf.float32)/tf.constant(K_r,dtype=tf.float32)
        #print(self.weight)
        #n_gmm*dim*dim*K_b
        self.Sigma=tf.transpose(tf.matrix_diag_part(tf.transpose(tf.constant(Sigma),[3,2,0,1])),[0,2,1])
        #print(self.Prcs)
        #n_gmm*dim*K_b
        self.mean=tf.transpose(tf.constant(mean),[2,0,1])

        self.KL_m=tf.transpose(tf.constant(KL_m))



def cal_KL_symm_diag(mu_data,Sigma_data,Prcs_data,weight_data,mu_hat,Prcs_hat,weight_hat):
    q_ce=cal_variation_q_diag(mu_data,Sigma_data,mu_hat,Prcs_hat,weight_hat,1.0)
    # print(q_ce)
    q_se=cal_variation_q_diag(mu_data,Sigma_data,mu_data,Prcs_data,weight_data,1.0)
    ce=cal_entropy_diag(weight_data,q_ce,mu_data,Sigma_data,weight_hat,mu_hat,Prcs_hat)
    se=cal_entropy_diag(weight_data,q_se,mu_data,Sigma_data,weight_data,mu_data,Prcs_data)
    kl=se-ce
    return kl



def cal_variation_q_diag(mu_data,Sigma_data,mu_hat,Prcs_hat,weight_hat,N_v):
    K_b=mu_data.shape[1]
    K_r=mu_hat.shape[1]
    q=np.zeros((K_b,K_r))
    for j in range(K_r):
        q[j,:]=cal_q_kl_diag(mu_data[:,j],Sigma_data[:,j],mu_hat,Prcs_hat,weight_hat,N_v)
        # print(q[j,:])
    return q


def cal_entropy_diag(weight_data,q,mu_data,Sigma_data,weight_hat,mu_hat,Prcs_hat):
    ent=0
    K_r=mu_hat.shape[1]
    [dim,K_b]=mu_data.shape
    for j_b in range(K_b):
        l_s=0
        for j_r in range(K_r):
            if q[j_b,j_r]!=0.0:
                t=np.sum(np.log(Prcs_hat[:,j_r]))-np.sum(np.multiply(np.multiply(np.transpose(mu_data[:,j_b]-mu_hat[:,j_r]),Prcs_hat[:,j_r]),mu_data[:,j_b]-mu_hat[:,j_r]))- \
                  np.sum(np.multiply(Prcs_hat[:, j_r], Sigma_data[:,j_b])) + 2*(np.log(weight_hat[j_r])-np.log(q[j_b,j_r])) - dim * np.log(2 * np.pi)
                l_s+=q[j_b,j_r]*t
        ent+=l_s*weight_data[j_b]
    ent=ent/2
    return ent

def cal_q_kl_diag(mu_k,Sigma_k,mu_hat,Prcs_hat,weight_hat,N_v):
    [dim,K_r]=mu_hat.shape
    l=np.zeros((K_r,1))
    for j in range(K_r):
        temp=np.sum(np.log(Prcs_hat[:,j]))-np.sum(np.multiply(np.multiply(np.transpose(mu_k-mu_hat[:,j]),Prcs_hat[:,j]),mu_k-mu_hat[:,j]))-\
             np.sum(np.multiply(Prcs_hat[:,j],Sigma_k))-dim*np.log(2*np.pi)
        l[j,0]=np.log(weight_hat[j])+0.5*N_v*temp
    l_max=np.max(l)
    l_minux_max=l-l_max
    rob_l=l_max+np.log(np.sum(np.exp(l_minux_max)))
    return np.reshape(np.exp(l-rob_l),(K_r,))



def cal_KL_symm(mu_data,Sigma_data,Prcs_data,weight_data,mu_hat,Prcs_hat,weight_hat):
    q_ce=cal_variation_q(mu_data,Sigma_data,mu_hat,Prcs_hat,weight_hat,1.0)
    q_se=cal_variation_q(mu_data,Sigma_data,mu_data,Prcs_data,weight_data,1.0)
    ce=cal_entropy(weight_data,q_ce,mu_data,Sigma_data,weight_hat,mu_hat,Prcs_hat)
    se=cal_entropy(weight_data,q_se,mu_data,Sigma_data,weight_data,mu_data,Prcs_data)
    kl=se-ce
    return kl

def cal_variation_q(mu_data,Sigma_data,mu_hat,Prcs_hat,weight_hat,N_v):
    K_b=mu_data.shape[1]
    K_r=mu_hat.shape[1]
    q=np.zeros((K_b,K_r))
    for j in range(K_b):
        q[j,:]=cal_q_kl(mu_data[:,j],Sigma_data[:,:,j],mu_hat,Prcs_hat,weight_hat,N_v)
        # print(q[j,:])
    return q


def cal_entropy(weight_data,q,mu_data,Sigma_data,weight_hat,mu_hat,Prcs_hat):
    ent=0
    K_r=mu_hat.shape[1]
    [dim,K_b]=mu_data.shape
    for j_b in range(K_b):
        l_s=0
        for j_r in range(K_r):
            mu_b=np.reshape(mu_data[:,j_b],(dim,1))
            mu_r=np.reshape(mu_hat[:,j_r],(dim,1))
            Prcs_r=np.reshape(Prcs_hat[:,:,j_r],(dim,dim))
            Sigma_b=np.reshape(Sigma_data[:,:,j_b],(dim,dim))
            if q[j_b,j_r] != 0.0:
                t=np.log(np.linalg.det(Prcs_r))- np.matmul(np.matmul(np.transpose(mu_b-mu_r),Prcs_r),mu_b-mu_r) - \
                  np.trace(np.matmul(Prcs_r,Sigma_b)) + 2*(np.log(weight_hat[j_r])-np.log(q[j_b,j_r])) - dim * np.log(2 * np.pi)
                l_s+=q[j_b,j_r]*t
        ent+=l_s*weight_data[j_b]
    ent=ent/2.0
    return ent

def cal_q_kl(mu_k,Sigma_k,mu_hat,Prcs_hat,weight_hat,N_v):
    [dim,K_r]=mu_hat.shape
    mu_k_k=np.reshape(mu_k,(dim,1))
    l=np.zeros((K_r,1))
    for j in range(K_r):
        mu_r = np.reshape(mu_hat[:, j], (dim, 1))
        Prcs_r = np.reshape(Prcs_hat[:, :, j], (dim, dim))
        temp=np.log(np.linalg.det(Prcs_r))-np.matmul(np.matmul(np.transpose(mu_k_k-mu_r),Prcs_r),mu_k_k-mu_r)-\
             np.trace(np.matmul(Prcs_r,Sigma_k))-dim*np.log(2*np.pi)
        l[j,0]=np.log(weight_hat[j])+0.5*N_v*temp
    soft_l=softmax(l)
    return np.reshape(soft_l,(K_r,))
