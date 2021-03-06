#this defines the loss function based on w,y,z and manifold parameters,
#there should be 2 models, one for w,y,z, the other for manifold parameters

import numpy as np
import tensorflow as tf
from GMM_para import GaussianMixture_LS
from GMM_para import GaussianMixture_HLS
from GMM_para import GaussianMixture_KHLS
from GMM_para import GaussianMixture_KHLS_NN
from GMM_para import GaussianMixture_HLS_diag_cov
from GMM_para import GaussianMixture_KHLS_diag_cov


class Model_LS():
    def __init__(self, sess, dim,n_gmm, K_r, K_b, d_w, d_y, d_z, n_v, p_model,scope_name='inverse_KL'):
        self.p_model=p_model
        self.K_r=K_r
        self.K_b=K_b
        self.n_gmm=n_gmm
        self.n_v=n_v
        self.sess=sess
        with tf.variable_scope(scope_name) as scope:
            self.w = tf.get_variable('w', shape=(n_gmm, d_w), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.y = tf.get_variable('y', shape=(n_gmm, d_y), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.z=tf.get_variable('z', shape=(n_gmm, d_z), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.b = tf.get_variable('b', shape=(dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.mu = tf.get_variable('mu', shape=(d_z,dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 100.0))
            self.C=tf.get_variable('C', shape=(d_y,dim,dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 10.0))
            self.beta=tf.get_variable('beta',shape=(1, 1, 1, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.xi = tf.get_variable('xi', shape=(d_w, K_r), dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(0.0, 1.0))
            #print(self.w)

            self.q_approx=GaussianMixture_LS(self.w,self.y,self.z,self.b,self.beta,self.C,self.mu,self.xi,n_gmm,dim,K_r,K_b)

            #n_gmm*K_b*K_r
            #psi_var_init=tf.multiply(tf.tile(tf.expand_dims(self.init_weight,axis=1),[1,self.K_b,1]),
            #                         tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]))
            #psi_var_init = tf.constant(np.ones((self.n_gmm,self.K_b,self.K_r),dtype=np.float32)/(np.float32(self.K_r)*np.float32(self.K_b)),dtype=tf.float32)
            #print(psi_var_init)
        q_var_init=tf.constant(np.ones((self.n_gmm,self.K_b,self.K_r),dtype=np.float32)/(np.float32(self.K_r)),dtype=tf.float32)
        self.q_v=tf.get_variable("phi", dtype=tf.float32, initializer=q_var_init)

        self.lv_train=[self.y,self.z]

        self.mu_train=[self.mu,self.b]

        self.Sigma_train=[self.C,self.beta]

        self.weight_train=[self.w,self.xi]

        self.lv_loss=self.get_lv_loss()

        self.mu_loss=self.get_mu_loss()

        self.Sigma_loss = self.get_Sigma_loss()

        self.weight_loss=self.get_weight_loss()



        # self.q_v=self.update_q_v()


        tf.summary.scalar('loss',self.lv_loss)


    def get_lv_loss(self):
        # calculate trace term: n_gmm*K_b*K_r
        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(self.q_approx.Prcs,axis=3), tf.expand_dims(self.p_model.Sigma,axis=4)),[1,2])

        # mu diff n_gmm*dim*K_b*K_r
        mu_diff=tf.subtract(tf.expand_dims(self.q_approx.mean,axis=2),tf.expand_dims(self.p_model.mean,axis=3))

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),[1,2])

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*self.q_approx.logdet_q


        clip_q_weight=tf.clip_by_value(self.q_approx.weight,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)




        #loss:
        loss_lv=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat))

        return loss_lv





    def get_mu_loss(self):
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)), [1, 2])
        # mu diff n_gmm*dim*K_b*K_r
        mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

        # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs = 0.5 * self.q_approx.logdet_q
        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        loss_mu = tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs+tf.stop_gradient(kl_trace-logdet_prcs-log_pi_hat)))
        return loss_mu

    def get_weight_loss(self):

        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        # loss:
        loss_weight = -tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), log_pi_hat))

        return loss_weight

    def get_Sigma_loss(self):
            # calculate trace term: n_gmm*K_b*K_r
            kl_trace = 0.5 * tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)),
                [1, 2])
            #print(kl_trace)
            # mu diff n_gmm*dim*K_b*K_r
            mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

            # n_gmm * dim *dim* K_b * K_r
            mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
            expd_Prcs = tf.tile(tf.expand_dims(self.q_approx.Prcs, axis=3), [1, 1, 1, self.K_b, 1])

            # mu^T*Prcs*mu term: n_gmm*K_b*K_r
            mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

            # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
            logdet_prcs = 0.5 * self.q_approx.logdet_q
            clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
            log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])
            #q_v_tensor=tf.constant(q_v_var)
            #print_q_v=tf.Print(self.q_v,[self.q_v])
            pi_q_v = tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight, axis=2), [1, 1, self.K_r]), self.q_v)

            # loss:
            loss_Sigma = tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs + kl_trace - logdet_prcs-log_pi_hat))

            return loss_Sigma

    def update_q_v(self):
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)), [1, 2])
        # mu diff n_gmm*dim*K_b*K_r
        mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs = tf.tile(tf.expand_dims(self.q_approx.Prcs, axis=3), [1, 1, 1, self.K_b, 1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

        # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs = 0.5 * self.q_approx.logdet_q

        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        log_term=log_pi_hat+tf.multiply(self.n_v,(logdet_prcs-mu_diff_prcs-kl_trace))
        assignment=self.q_v.assign(tf.nn.softmax(tf.stop_gradient(log_term),axis=2))

        update_q=self.sess.run(assignment)
        return update_q
        #print(update_q[0,0,:])







class Model_KHLS():
    def __init__(self, sess, config, dim,n_gmm, K_r, K_b, d_v,d_w, d_y, d_z, n_v, n_ps_v,p_model,scope_name='inverse_KL'):
        self.p_model=p_model
        self.K_r=K_r
        self.K_b=K_b
        self.dim=dim
        self.n_gmm=n_gmm
        self.n_v=n_v
        self.sess=sess
        self.config=config
        self.n_ps_v=n_ps_v
        with tf.variable_scope(scope_name) as scope:
            self.v=tf.get_variable('v',shape=(n_gmm,d_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.ps_v=tf.get_variable('ps_v',shape=(n_ps_v,d_v),dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(0.0,1.0))
            self.H_w=tf.get_variable('H_w',shape=(d_w,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.H_z=tf.get_variable('H_z',shape=(d_z,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.H_y=tf.get_variable('H_y',shape=(d_y,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))

            self.b = tf.get_variable('b', shape=(dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.mu = tf.get_variable('mu', shape=(d_z,dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 100.0))
            self.C=tf.get_variable('C', shape=(d_y,dim,dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.beta=tf.get_variable('beta',shape=(1, 1, 1, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.xi = tf.get_variable('xi', shape=(d_w, K_r), dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(0.0, 1.0))
            #print(self.w)

            self.q_approx=GaussianMixture_KHLS(self.v,self.ps_v, self.n_ps_v, self.config.lamda_kern, self.config.gamma_width, self.config.gamma_scale, self.config.poly_scale,
                                               self.H_w,self.H_y,self.H_z,self.b,self.beta,self.C,self.mu,self.xi,n_gmm,dim,K_r,K_b)

            #n_gmm*K_b*K_r
            #psi_var_init=tf.multiply(tf.tile(tf.expand_dims(self.init_weight,axis=1),[1,self.K_b,1]),
            #                         tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]))
            #psi_var_init = tf.constant(np.ones((self.n_gmm,self.K_b,self.K_r),dtype=np.float32)/(np.float32(self.K_r)*np.float32(self.K_b)),dtype=tf.float32)
            #print(psi_var_init)
        q_cons=np.zeros((self.n_gmm,self.K_b,self.K_r))
        for i in range(n_gmm):
            for j_o in range(K_b):
                q_cons[i,j_o,j_o]=1.0
        q_var_init=tf.constant(q_cons,dtype=tf.float32)

        # q_var_init=tf.constant(np.ones((self.n_gmm,self.K_b,self.K_r),dtype=np.float32)/(np.float32(self.K_r)),dtype=tf.float32)
        self.q_v=tf.get_variable("phi", dtype=tf.float32, initializer=q_var_init)

        self.lv_train=[self.v, self.ps_v, self.H_w, self.H_y, self.H_z]

        self.mu_train=[self.mu,self.b]

        self.Sigma_train=[self.C,self.beta]

        self.weight_train=self.xi

        self.lv_loss=self.get_lv_loss()

        self.mu_loss=self.get_mu_loss()

        self.Sigma_loss = self.get_Sigma_loss()

        self.weight_loss=self.get_weight_loss()



        tf.summary.scalar('loss',self.lv_loss)


    def get_lv_loss(self):
        # calculate trace term: n_gmm*K_b*K_r
        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(self.q_approx.Prcs,axis=3), tf.expand_dims(self.p_model.Sigma,axis=4)),[1,2])

        # mu diff n_gmm*dim*K_b*K_r
        mu_diff=tf.subtract(tf.expand_dims(self.q_approx.mean,axis=2),tf.expand_dims(self.p_model.mean,axis=3))
        #print(mu_diff)

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),[1,2])

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*self.q_approx.logdet_q


        clip_q_weight=tf.clip_by_value(self.q_approx.weight,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        v_norm_mat=tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(self.v,axis=0),tf.expand_dims(self.v,axis=1))),axis=2)
        # print(v_norm_mat)
        # print(self.p_model.KL_m)

        #loss:
        loss_lv=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat))+\
                self.config.reg_v*tf.reduce_sum(tf.square(self.v))+self.config.reg_wyz*(tf.reduce_sum(tf.square(self.q_approx.y))+tf.reduce_sum(tf.square(self.q_approx.w))+tf.reduce_sum(tf.square(self.q_approx.z)))\
                +self.config.reg_kl*tf.reduce_sum(tf.square(v_norm_mat-self.p_model.KL_m))
        # loss_lv=self.config.reg_kl*tf.reduce_sum(tf.square(v_norm_mat-self.p_model.KL_m))
        return loss_lv





    def get_mu_loss(self):
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)), [1, 2])
        # mu diff n_gmm*dim*K_b*K_r
        mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

        # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs = 0.5 * self.q_approx.logdet_q
        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        loss_mu = tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs+tf.stop_gradient(kl_trace-logdet_prcs-log_pi_hat)))
        return loss_mu

    def get_weight_loss(self):

        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        # loss:
        loss_weight = -tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), log_pi_hat))

        return loss_weight

    def get_Sigma_loss(self):
            # calculate trace term: n_gmm*K_b*K_r
            kl_trace = 0.5 * tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)),
                [1, 2])
            #print(kl_trace)
            # mu diff n_gmm*dim*K_b*K_r
            mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

            # n_gmm * dim *dim* K_b * K_r
            mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
            expd_Prcs = tf.tile(tf.expand_dims(self.q_approx.Prcs, axis=3), [1, 1, 1, self.K_b, 1])

            # mu^T*Prcs*mu term: n_gmm*K_b*K_r
            mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

            # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
            logdet_prcs = 0.5 * self.q_approx.logdet_q
            clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
            log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])
            #q_v_tensor=tf.constant(q_v_var)
            #print_q_v=tf.Print(self.q_v,[self.q_v])
            pi_q_v = tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight, axis=2), [1, 1, self.K_r]), self.q_v)

            # loss:
            loss_Sigma = tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs + kl_trace - logdet_prcs-log_pi_hat))

            return loss_Sigma

    def update_q_v(self):
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)), [1, 2])
        # mu diff n_gmm*dim*K_b*K_r
        mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs = tf.tile(tf.expand_dims(self.q_approx.Prcs, axis=3), [1, 1, 1, self.K_b, 1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

        # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs = 0.5 * self.q_approx.logdet_q

        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        log_term=log_pi_hat+tf.multiply(self.n_v,(logdet_prcs-mu_diff_prcs-kl_trace))
        assignment=self.q_v.assign(tf.nn.softmax(tf.stop_gradient(log_term),axis=2))

        update_q=self.sess.run(assignment)
        return update_q

    def MH_sample(self):
        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(self.q_approx.Prcs,axis=3), tf.expand_dims(self.p_model.Sigma,axis=4)),[1,2])
        mu_diff=tf.subtract(tf.expand_dims(self.q_approx.mean,axis=2),tf.expand_dims(self.p_model.mean,axis=3))

        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),[1,2])

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*self.q_approx.logdet_q


        clip_q_weight=tf.clip_by_value(self.q_approx.weight,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        kl_arr=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat),axis=[1,2])
        rand_num=tf.random.uniform([])
        sample_ind=tf.cond(rand_num>0.8, lambda : tf.random.uniform([],maxval=self.n_gmm-1,dtype=tf.dtypes.int64),lambda : tf.math.argmax(kl_arr))

        q_sample_before=tf.slice(self.q_v, [sample_ind, 0, 0], [1, self.K_b, self.K_r])

        return sample_ind,kl_arr,q_sample_before

    def MH_sample_update(self,sample,kl_arr_val,q_swap,it):
        q_sample_before=tf.slice(self.q_v, [sample, 0, 0], [1, self.K_b, self.K_r])

        #shuffle the q
        q_sample=tf.constant(q_swap)
        # print(q_sample)
        # print(q_sample)
        # sample from reconstructed GMMs
        mu_hat=tf.slice(self.q_approx.mean,[sample,0,0],[1,self.dim,self.K_r])
        Prcs_hat=tf.slice(self.q_approx.Prcs,[sample,0,0,0],[1,self.dim,self.dim,self.K_r])
        weight_hat=tf.slice(self.q_approx.weight,[sample,0],[1,self.K_r])
        logdet_hat=tf.slice(self.q_approx.logdet_q,[sample,0,0],[1,self.K_b,self.K_r])
        #sample from GT GMMs
        mu_data=tf.slice(self.p_model.mean,[sample,0,0],[1,self.dim,self.K_b])
        Sigma_data=tf.slice(self.p_model.Sigma,[sample,0,0,0],[1,self.dim,self.dim,self.K_b])
        weight_data=tf.slice(self.p_model.weight,[sample,0],[1,self.K_b])

        # calculate the KL

        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(Prcs_hat,axis=3), tf.expand_dims(Sigma_data,axis=4)),[1,2])
        mu_diff=tf.subtract(tf.expand_dims(mu_hat,axis=2),tf.expand_dims(mu_data,axis=3))

        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(Prcs_hat,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),[1,2])

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*logdet_hat


        clip_q_weight=tf.clip_by_value(weight_hat,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(weight_data,axis=2),[1,1,self.K_r]),q_sample)


        kl_sample=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat))

        rand_num=tf.random.uniform([])

        T=1/(self.config.T_coeff*np.log(self.config.T_0+it-self.config.start_MC_q))

        prob_q=tf.dtypes.cast(tf.exp(-kl_arr_val[sample,]/T),dtype=tf.float32)

        prob_q_sample=tf.exp(-kl_sample/T)

        # print(prob_q)
        # print(prob_q_sample)

        # if prob_q_sample>prob_q, then replace the q
        # if prob_q_sample<prob_q, then replace the q with probability prob_q_sample/prob_q

        swap_prob=tf.math.minimum(prob_q_sample/prob_q,1.0)

        print_swap_prob=tf.Print(swap_prob,[swap_prob])
        q_sample_MH=tf.cond(rand_num>print_swap_prob,lambda :q_sample_before, lambda : q_sample)

        # put q_sample_MH back to self.q_v
        assignment_q_sample=self.q_v[sample,:,:].assign(tf.squeeze(q_sample_MH))

        return assignment_q_sample

class Model_KHLS_NN():
    def __init__(self, sess, config, dim,n_gmm, K_r, K_b, d_v,d_w, d_y, d_z, d_mu_1,n_v, n_ps_v,p_model,scope_name='inverse_KL'):
        self.p_model=p_model
        self.K_r=K_r
        self.K_b=K_b
        self.dim=dim
        self.n_gmm=n_gmm
        self.n_v=n_v
        self.sess=sess
        self.config=config
        self.n_ps_v=n_ps_v
        with tf.variable_scope(scope_name) as scope:
            self.v=tf.get_variable('v',shape=(n_gmm,d_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.ps_v=tf.get_variable('ps_v',shape=(n_ps_v,d_v),dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(0.0,1.0))
            self.H_w=tf.get_variable('H_w',shape=(d_w,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.H_z=tf.get_variable('H_z',shape=(d_z,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.H_y=tf.get_variable('H_y',shape=(d_y,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))

            self.mu_1=tf.get_variable('mu_1',shape=(d_z,d_mu_1,K_r),dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(0.0,1.0))
            self.b_1=tf.get_variable('b_1',shape=(d_mu_1,K_r),dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0,1.0))

            self.mu_out=tf.get_variable('mu_out',shape=(d_mu_1,dim,K_r),dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(0.0,1.0))
            self.b_out=tf.get_variable('b_out',shape=(1,dim,K_r),dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0,1.0))

            self.C=tf.get_variable('C', shape=(d_y,dim,dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.beta=tf.get_variable('beta',shape=(1, 1, 1, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.xi = tf.get_variable('xi', shape=(d_w, K_r), dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(0.0, 1.0))
            #print(self.w)

            self.q_approx=GaussianMixture_KHLS_NN(self.v,self.ps_v, self.n_ps_v, self.config.lamda_kern, self.config.gamma_width, self.config.gamma_scale, self.config.poly_scale,
                                               self.H_w,self.H_y,self.H_z,self.beta,self.C,self.mu_1,self.b_1,self.mu_out,self.b_out,self.xi,n_gmm,dim,K_r,K_b)

            #n_gmm*K_b*K_r
            #psi_var_init=tf.multiply(tf.tile(tf.expand_dims(self.init_weight,axis=1),[1,self.K_b,1]),
            #                         tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]))
            #psi_var_init = tf.constant(np.ones((self.n_gmm,self.K_b,self.K_r),dtype=np.float32)/(np.float32(self.K_r)*np.float32(self.K_b)),dtype=tf.float32)
            #print(psi_var_init)
        # q_cons=np.zeros((self.n_gmm,self.K_b,self.K_r))
        # for i in range(n_gmm):
        #     for j_o in range(K_b):
        #         q_cons[i,j_o,j_o]=1.0
        # q_var_init=tf.constant(q_cons,dtype=tf.float32)

        q_var_init=tf.constant(np.ones((self.n_gmm,self.K_b,self.K_r),dtype=np.float32)/(np.float32(self.K_r)),dtype=tf.float32)
        self.q_v=tf.get_variable("phi", dtype=tf.float32, initializer=q_var_init)

        self.lv_train=[self.v, self.ps_v, self.H_w, self.H_y, self.H_z]

        self.mu_train=[self.mu_1,self.b_1,self.mu_out,self.b_out]

        self.Sigma_train=[self.C,self.beta]

        self.weight_train=self.xi

        self.lv_loss=self.get_lv_loss()

        self.mu_loss=self.get_mu_loss()

        self.Sigma_loss = self.get_Sigma_loss()

        self.weight_loss=self.get_weight_loss()



        tf.summary.scalar('loss',self.lv_loss)


    def get_lv_loss(self):
        # calculate trace term: n_gmm*K_b*K_r
        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(self.q_approx.Prcs,axis=3), tf.expand_dims(self.p_model.Sigma,axis=4)),[1,2])

        # mu diff n_gmm*dim*K_b*K_r
        mu_diff=tf.subtract(tf.expand_dims(self.q_approx.mean,axis=2),tf.expand_dims(self.p_model.mean,axis=3))
        #print(mu_diff)

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),[1,2])

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*self.q_approx.logdet_q


        clip_q_weight=tf.clip_by_value(self.q_approx.weight,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        v_norm_mat=tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(self.v,axis=0),tf.expand_dims(self.v,axis=1))),axis=2)
        # print(v_norm_mat)
        # print(self.p_model.KL_m)

        #loss:
        loss_lv=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat))+\
                self.config.reg_v*tf.reduce_sum(tf.square(self.v))+self.config.reg_wyz*(tf.reduce_sum(tf.square(self.q_approx.y))+tf.reduce_sum(tf.square(self.q_approx.w))+tf.reduce_sum(tf.square(self.q_approx.z)))\
                +self.config.reg_kl*tf.reduce_sum(tf.square(v_norm_mat-self.p_model.KL_m))
        # loss_lv=self.config.reg_kl*tf.reduce_sum(tf.square(v_norm_mat-self.p_model.KL_m))
        return loss_lv





    def get_mu_loss(self):
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)), [1, 2])
        # mu diff n_gmm*dim*K_b*K_r
        mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

        # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs = 0.5 * self.q_approx.logdet_q
        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        loss_mu = tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs+tf.stop_gradient(kl_trace-logdet_prcs-log_pi_hat)))
        return loss_mu

    def get_weight_loss(self):

        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        # loss:
        loss_weight = -tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), log_pi_hat))

        return loss_weight

    def get_Sigma_loss(self):
            # calculate trace term: n_gmm*K_b*K_r
            kl_trace = 0.5 * tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)),
                [1, 2])
            #print(kl_trace)
            # mu diff n_gmm*dim*K_b*K_r
            mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

            # n_gmm * dim *dim* K_b * K_r
            mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
            expd_Prcs = tf.tile(tf.expand_dims(self.q_approx.Prcs, axis=3), [1, 1, 1, self.K_b, 1])

            # mu^T*Prcs*mu term: n_gmm*K_b*K_r
            mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

            # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
            logdet_prcs = 0.5 * self.q_approx.logdet_q
            clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
            log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])
            #q_v_tensor=tf.constant(q_v_var)
            #print_q_v=tf.Print(self.q_v,[self.q_v])
            pi_q_v = tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight, axis=2), [1, 1, self.K_r]), self.q_v)

            # loss:
            loss_Sigma = tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs + kl_trace - logdet_prcs-log_pi_hat))

            return loss_Sigma

    def update_q_v(self):
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=3), tf.expand_dims(self.p_model.Sigma, axis=4)), [1, 2])
        # mu diff n_gmm*dim*K_b*K_r
        mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

        # n_gmm * dim *dim* K_b * K_r
        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs = tf.tile(tf.expand_dims(self.q_approx.Prcs, axis=3), [1, 1, 1, self.K_b, 1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), [1, 2])

        # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs = 0.5 * self.q_approx.logdet_q

        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        log_term=log_pi_hat+tf.multiply(self.n_v,(logdet_prcs-mu_diff_prcs-kl_trace))
        assignment=self.q_v.assign(tf.nn.softmax(tf.stop_gradient(log_term),axis=2))

        update_q=self.sess.run(assignment)
        return update_q

    def MH_sample(self):
        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(self.q_approx.Prcs,axis=3), tf.expand_dims(self.p_model.Sigma,axis=4)),[1,2])
        mu_diff=tf.subtract(tf.expand_dims(self.q_approx.mean,axis=2),tf.expand_dims(self.p_model.mean,axis=3))

        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),[1,2])

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*self.q_approx.logdet_q


        clip_q_weight=tf.clip_by_value(self.q_approx.weight,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        kl_arr=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat),axis=[1,2])
        rand_num=tf.random.uniform([])
        sample_ind=tf.cond(rand_num>0.5, lambda : tf.random.uniform([],maxval=self.n_gmm-1,dtype=tf.dtypes.int64),lambda : tf.math.argmax(kl_arr))

        q_sample_before=tf.slice(self.q_v, [sample_ind, 0, 0], [1, self.K_b, self.K_r])

        return sample_ind,kl_arr,q_sample_before

    def MH_sample_update(self,sample,kl_arr_val,q_swap,it):
        q_sample_before=tf.slice(self.q_v, [sample, 0, 0], [1, self.K_b, self.K_r])

        #shuffle the q
        q_sample=tf.constant(q_swap)
        # print(q_sample)
        # print(q_sample)
        # sample from reconstructed GMMs
        mu_hat=tf.slice(self.q_approx.mean,[sample,0,0],[1,self.dim,self.K_r])
        Prcs_hat=tf.slice(self.q_approx.Prcs,[sample,0,0,0],[1,self.dim,self.dim,self.K_r])
        weight_hat=tf.slice(self.q_approx.weight,[sample,0],[1,self.K_r])
        logdet_hat=tf.slice(self.q_approx.logdet_q,[sample,0,0],[1,self.K_b,self.K_r])
        #sample from GT GMMs
        mu_data=tf.slice(self.p_model.mean,[sample,0,0],[1,self.dim,self.K_b])
        Sigma_data=tf.slice(self.p_model.Sigma,[sample,0,0,0],[1,self.dim,self.dim,self.K_b])
        weight_data=tf.slice(self.p_model.weight,[sample,0],[1,self.K_b])

        # calculate the KL

        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(Prcs_hat,axis=3), tf.expand_dims(Sigma_data,axis=4)),[1,2])
        mu_diff=tf.subtract(tf.expand_dims(mu_hat,axis=2),tf.expand_dims(mu_data,axis=3))

        mu_diff_dot = tf.multiply(tf.expand_dims(mu_diff, axis=1), tf.expand_dims(mu_diff, axis=2))
        expd_Prcs=tf.tile(tf.expand_dims(Prcs_hat,axis=3),[1,1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),[1,2])

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*logdet_hat


        clip_q_weight=tf.clip_by_value(weight_hat,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(weight_data,axis=2),[1,1,self.K_r]),q_sample)


        kl_sample=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat))

        rand_num=tf.random.uniform([])

        T=1/(self.config.T_coeff*np.log(self.config.T_0+it-self.config.start_MC_q))

        prob_q=tf.dtypes.cast(tf.exp(-kl_arr_val[sample,]/T),dtype=tf.float32)

        prob_q_sample=tf.exp(-kl_sample/T)

        # print(prob_q)
        # print(prob_q_sample)

        # if prob_q_sample>prob_q, then replace the q
        # if prob_q_sample<prob_q, then replace the q with probability prob_q_sample/prob_q

        swap_prob=tf.math.minimum(prob_q_sample/prob_q,1.0)

        print_swap_prob=tf.Print(swap_prob,[swap_prob])
        q_sample_MH=tf.cond(rand_num>print_swap_prob,lambda :q_sample_before, lambda : q_sample)

        # put q_sample_MH back to self.q_v
        assignment_q_sample=self.q_v[sample,:,:].assign(tf.squeeze(q_sample_MH))

        return assignment_q_sample



class Model_KHLS_diag_cov():
    def __init__(self, sess, config,dim,n_gmm, K_r, K_b, d_v,d_w, d_y, d_z, d_C_1, n_v, n_ps_v,p_model,scope_name='inverse_KL'):
        self.p_model=p_model
        self.K_r=K_r
        self.K_b=K_b
        self.dim=dim
        self.n_gmm=n_gmm
        self.n_v=n_v
        self.sess=sess
        self.config=config
        self.n_ps_v=n_ps_v
        with tf.variable_scope(scope_name) as scope:
            self.v=tf.get_variable('v',shape=(n_gmm,d_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.ps_v=tf.get_variable('ps_v',shape=(n_ps_v,d_v),dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(0.0,1.0))
            self.H_w=tf.get_variable('H_w',shape=(d_w,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.H_z=tf.get_variable('H_z',shape=(d_z,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.H_y=tf.get_variable('H_y',shape=(d_y,n_ps_v),dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))

            self.b = tf.get_variable('b', shape=(dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 1.0))
            self.mu = tf.get_variable('mu', shape=(d_z,dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 100.0))


            self.C_1=tf.get_variable('C_1', shape=(d_y,d_C_1, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 0.5))
            self.C_b_1=tf.get_variable('C_b_1', shape=(1,d_C_1, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 0.5))


            # self.C_2=tf.get_variable('C_2', shape=(d_C_1,d_C_2, K_r), dtype=tf.float32,
            #                          initializer=tf.random_uniform_initializer(0.0, 1.0))
            # self.C_b_2=tf.get_variable('C_b_2', shape=(1,d_C_2, K_r), dtype=tf.float32,
            #                          initializer=tf.random_uniform_initializer(0.0, 1.0))


            self.C_out=tf.get_variable('C_out', shape=(d_C_1,dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 0.5))
            self.C_out_b=tf.get_variable('C_out_b', shape=(1,dim, K_r), dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0.0, 0.5))



            self.xi = tf.get_variable('xi', shape=(d_w, K_r), dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(0.0, 1.0))
            #print(self.w)
            self.q_approx=GaussianMixture_KHLS_diag_cov(self.v,self.ps_v, self.n_ps_v, self.config.lamda_kern, self.config.gamma_width, self.config.gamma_scale, self.config.poly_scale,
                                                        self.H_w,self.H_y,self.H_z,self.b,self.C_1,self.C_b_1,self.C_out,self.C_out_b,self.mu,self.xi,n_gmm,dim,K_r,K_b)

        q_var_init=tf.constant(np.ones((self.n_gmm,self.K_b,self.K_r),dtype=np.float32)/(np.float32(self.K_r)),dtype=tf.float32)

        self.q_v=tf.get_variable("phi", dtype=tf.float32, initializer=q_var_init)

        self.lv_train=[self.v, self.ps_v, self.H_w, self.H_y, self.H_z]

        self.mu_train=[self.mu,self.b]

        self.Sigma_train=[self.C_1,self.C_b_1,self.C_out,self.C_out_b]

        self.weight_train=self.xi

        self.lv_loss=self.get_lv_loss()

        self.mu_loss=self.get_mu_loss()

        self.Sigma_loss = self.get_Sigma_loss()

        self.weight_loss=self.get_weight_loss()



        tf.summary.scalar('loss',self.lv_loss)


    def get_lv_loss(self):
        # calculate trace term: n_gmm*K_b*K_r
        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(self.q_approx.Prcs,axis=2), tf.expand_dims(self.p_model.Sigma,axis=3)),axis=1)

        # mu diff n_gmm*dim*K_b*K_r
        mu_diff=tf.subtract(tf.expand_dims(self.q_approx.mean,axis=2),tf.expand_dims(self.p_model.mean,axis=3))

        # n_gmm * dim *K_b * K_r
        mu_diff_dot=tf.square(mu_diff)
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=2),[1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),axis=1)

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*self.q_approx.logdet_q


        clip_q_weight=tf.clip_by_value(self.q_approx.weight,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        v_norm_mat=tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(self.v,axis=0),tf.expand_dims(self.v,axis=1))),axis=2)
        # print(v_norm_mat)
        # print(self.p_model.KL_m)

        #loss:
        loss_lv=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat))+\
                self.config.reg_v*tf.reduce_sum(tf.square(self.v))+self.config.reg_wyz*(tf.reduce_sum(tf.square(self.q_approx.y))+tf.reduce_sum(tf.square(self.q_approx.w))+tf.reduce_sum(tf.square(self.q_approx.z)))\
                +self.config.reg_kl*tf.reduce_sum(tf.square(v_norm_mat-self.p_model.KL_m))

        return loss_lv





    def get_mu_loss(self):
        kl_trace=0.5*tf.reduce_sum(tf.multiply(tf.expand_dims(self.q_approx.Prcs,axis=2), tf.expand_dims(self.p_model.Sigma,axis=3)),axis=1)

        # mu diff n_gmm*dim*K_b*K_r
        mu_diff=tf.subtract(tf.expand_dims(self.q_approx.mean,axis=2),tf.expand_dims(self.p_model.mean,axis=3))
        # n_gmm * dim *K_b * K_r
        mu_diff_dot=tf.square(mu_diff)
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=2),[1,1,self.K_b,1])
        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs=0.5*tf.reduce_sum(tf.multiply(mu_diff_dot,expd_Prcs),axis=1)

        # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs = 0.5 * self.q_approx.logdet_q
        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        loss_mu = tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs+tf.stop_gradient(kl_trace-logdet_prcs-log_pi_hat)))
        return loss_mu

    def get_weight_loss(self):

        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)

        # loss:
        loss_weight = -tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), log_pi_hat))

        return loss_weight

    def get_Sigma_loss(self):
            # calculate trace term: n_gmm*K_b*K_r
            kl_trace = 0.5 * tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=2), tf.expand_dims(self.p_model.Sigma, axis=3)),
                axis=1)

            # mu diff n_gmm*dim*K_b*K_r
            mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

            # n_gmm * dim *K_b * K_r
            mu_diff_dot = tf.square(mu_diff)
            expd_Prcs = tf.tile(tf.expand_dims(self.q_approx.Prcs, axis=2), [1, 1, self.K_b, 1])

            # mu^T*Prcs*mu term: n_gmm*K_b*K_r
            mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), axis=1)

            # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
            logdet_prcs = 0.5 * self.q_approx.logdet_q

            clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
            log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])
            #q_v_tensor=tf.constant(q_v_var)
            #print_q_v=tf.Print(self.q_v,[self.q_v])
            pi_q_v = tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight, axis=2), [1, 1, self.K_r]), self.q_v)

            # loss:
            loss_Sigma = tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs + kl_trace - logdet_prcs-log_pi_hat))

            return loss_Sigma

    def update_q_v(self):
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=2), tf.expand_dims(self.p_model.Sigma, axis=3)),
            axis=1)

        # mu diff n_gmm*dim*K_b*K_r
        mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

        # n_gmm * dim *K_b * K_r
        mu_diff_dot = tf.square(mu_diff)
        expd_Prcs = tf.tile(tf.expand_dims(self.q_approx.Prcs, axis=2), [1, 1, self.K_b, 1])
        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), axis=1)

        # logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs = 0.5 * self.q_approx.logdet_q

        clip_q_weight = tf.clip_by_value(self.q_approx.weight, 1e-30, 1.0)
        log_pi_hat = tf.tile(tf.expand_dims(tf.log(clip_q_weight), axis=1), [1, self.K_b, 1])

        log_term=log_pi_hat+tf.multiply(self.n_v,(logdet_prcs-mu_diff_prcs-kl_trace))
        assignment=self.q_v.assign(tf.nn.softmax(tf.stop_gradient(log_term),axis=2))

        update_q=self.sess.run(assignment)
        return update_q

    def MH_sample(self):
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.q_approx.Prcs, axis=2), tf.expand_dims(self.p_model.Sigma, axis=3)),
            axis=1)
        mu_diff = tf.subtract(tf.expand_dims(self.q_approx.mean, axis=2), tf.expand_dims(self.p_model.mean, axis=3))

        mu_diff_dot = tf.square(mu_diff)
        expd_Prcs=tf.tile(tf.expand_dims(self.q_approx.Prcs,axis=2),[1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), axis=1)

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*self.q_approx.logdet_q


        clip_q_weight=tf.clip_by_value(self.q_approx.weight,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(self.p_model.weight,axis=2),[1,1,self.K_r]),self.q_v)
        kl_arr=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v), mu_diff_prcs + kl_trace - logdet_prcs - log_pi_hat),axis=[1,2])
        rand_num=tf.random.uniform([])
        sample_ind=tf.cond(rand_num>0.8, lambda : tf.random.uniform([],maxval=self.n_gmm-1,dtype=tf.dtypes.int64),lambda : tf.math.argmax(kl_arr))

        q_sample_before=tf.slice(self.q_v, [sample_ind, 0, 0], [1, self.K_b, self.K_r])

        return sample_ind,kl_arr,q_sample_before

    def MH_sample_update(self,sample,kl_arr_val,q_swap,it):
        q_sample_before=tf.slice(self.q_v, [sample, 0, 0], [1, self.K_b, self.K_r])

        #shuffle the q
        q_sample=tf.constant(q_swap)
        # print(q_sample)
        # print(q_sample)
        # sample from reconstructed GMMs
        mu_hat=tf.slice(self.q_approx.mean,[sample,0,0],[1,self.dim,self.K_r])
        Prcs_hat=tf.slice(self.q_approx.Prcs,[sample,0,0],[1,self.dim,self.K_r])
        weight_hat=tf.slice(self.q_approx.weight,[sample,0],[1,self.K_r])
        logdet_hat=tf.slice(self.q_approx.logdet_q,[sample,0,0],[1,self.K_b,self.K_r])
        #sample from GT GMMs
        mu_data=tf.slice(self.p_model.mean,[sample,0,0],[1,self.dim,self.K_b])
        Sigma_data=tf.slice(self.p_model.Sigma,[sample,0,0],[1,self.dim,self.K_b])
        weight_data=tf.slice(self.p_model.weight,[sample,0],[1,self.K_b])

        # calculate the KL
        kl_trace = 0.5 * tf.reduce_sum(
            tf.multiply(tf.expand_dims(Prcs_hat, axis=2), tf.expand_dims(Sigma_data, axis=3)),
            axis=1)
        mu_diff = tf.subtract(tf.expand_dims(mu_hat, axis=2), tf.expand_dims(mu_data, axis=3))

        mu_diff_dot = tf.square(mu_diff)
        expd_Prcs=tf.tile(tf.expand_dims(Prcs_hat,axis=2),[1,1,self.K_b,1])

        # mu^T*Prcs*mu term: n_gmm*K_b*K_r
        mu_diff_prcs = 0.5 * tf.reduce_sum(tf.multiply(mu_diff_dot, expd_Prcs), axis=1)

        #logdet term n_gmm*K_b*K_r logdet_p=n_gmm*K_b
        logdet_prcs=0.5*logdet_hat


        clip_q_weight=tf.clip_by_value(weight_hat,1e-30,1.0)
        log_pi_hat=tf.tile(tf.expand_dims(tf.log(clip_q_weight),axis=1),[1,self.K_b,1])


        pi_q_v=tf.multiply(tf.tile(tf.expand_dims(weight_data,axis=2),[1,1,self.K_r]),q_sample)


        kl_sample=tf.reduce_sum(tf.multiply(tf.stop_gradient(pi_q_v),mu_diff_prcs+kl_trace-logdet_prcs-log_pi_hat))

        rand_num=tf.random.uniform([])

        T=1/(self.config.T_coeff*np.log(self.config.T_0+it-self.config.start_MC_q))

        prob_q=tf.dtypes.cast(tf.exp(-kl_arr_val[sample,]/T),dtype=tf.float32)

        prob_q_sample=tf.exp(-kl_sample/T)

        # print(prob_q)
        # print(prob_q_sample)

        # if prob_q_sample>prob_q, then replace the q
        # if prob_q_sample<prob_q, then replace the q with probability prob_q_sample/prob_q

        swap_prob=tf.math.minimum(prob_q_sample/prob_q,1.0)

        print_swap_prob=tf.Print(swap_prob,[swap_prob])
        q_sample_MH=tf.cond(rand_num>print_swap_prob,lambda :q_sample_before, lambda : q_sample)

        # put q_sample_MH back to self.q_v
        assignment_q_sample=self.q_v[sample,:,:].assign(tf.squeeze(q_sample_MH))

        return assignment_q_sample
