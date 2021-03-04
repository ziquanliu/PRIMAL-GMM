import tensorflow as tf

# from model_KHLS import Model_KHLS

from model_KHLS import Model_KHLS_NN

# from model_KHLS import Model_KHLS_diag_cov

from GMM_para import GaussianMixture_data

# from GMM_para import GaussianMixture_data_diag_cov

from GMM_para import cal_KL_symm_diag

# from GMM_para import cal_KL_symm

import scipy.io as sio

import numpy as np

import time

import h5py




class Trainer(object):
    def optimize_adam(self, loss, train_vars=None, lr=1e-2):
        optimizer = tf.train.AdamOptimizer(lr)
        if train_vars is None:
            train_op = optimizer.minimize(loss, global_step=self.global_step,
                                          gate_gradients=optimizer.GATE_NONE)
        else:
            train_op = optimizer.minimize(loss, var_list=train_vars,
                                          global_step=self.global_step,
                                          gate_gradients=optimizer.GATE_NONE)

        return train_op


    def optimize_adagrad(self, loss, train_vars=None, lr=1e-2):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9)  # adagrad with momentum
        if train_vars is None:
            train_op = optimizer.minimize(loss, global_step=self.global_step,
                                          gate_gradients=optimizer.GATE_NONE)
        else:
            train_op = optimizer.minimize(loss, var_list=train_vars,
                                          global_step=self.global_step,
                                          gate_gradients=optimizer.GATE_NONE)
        return train_op

    def __init__(self, config, session,mean_p,Sigma_p,weight_p,KL_m_norm,trial_i):
        self.config = config
        self.session = session


        dim=mean_p.shape[0]
        n_gmm=mean_p.shape[2]
        K_b=mean_p.shape[1]

        self.trial_i=trial_i




        # --- create model ---
        # self.p_target = GaussianMixture_data_diag_cov(mean_p,Sigma_p,weight_p,KL_m_norm)
        self.p_target=GaussianMixture_data(mean_p,Sigma_p,weight_p,KL_m_norm)

        # self.p_target = GaussianMixture_data_diag_cov(mean_p,Sigma_p,weight_p,KL_m_norm)
        # [mu_data,Sigma_data,weight_data]=self.session.run([self.p_target.mean,self.p_target.Sigma,self.p_target.weight])
        # Prcs_data = np.zeros((n_gmm,dim,K_b), dtype=np.float32)
        # for i in range(n_gmm):
        #     for j in range(K_b):
        #         Prcs_data[i,:,j] = np.linalg.inv(Sigma_data[i,:,j])
        # KL_mat=cal_KL_symm_diag(mu_data,Sigma_data,weight_data,Prcs_data)



        #initialize psi_var
        #self.n_temp=tf.Variable(2.0,name='temp')


        #dim, n_gmm, K_r, K_b, d_w, d_y, d_z, p_model, psi_var
        self.n_vs = config.n_vs
        # self.model = Model_KHLS(self.session,self.config,dim, n_gmm, self.config.K_r, K_b, self.config.d_v,self.config.d_w, self.config.d_y, self.config.d_z, self.n_vs,self.config.n_ps_v,self.p_target)
        self.model = Model_KHLS_NN(self.session,self.config,dim, n_gmm, self.config.K_r, K_b, self.config.d_v,self.config.d_w, self.config.d_y,self.config.d_z,self.config.d_mu_1, self.n_vs,self.config.n_ps_v,self.p_target)

        # self.model = Model_HLS_diag_cov(self.session,self.config,dim, n_gmm, self.config.K_r, K_b, self.config.d_v,self.config.d_w, self.config.d_y, self.config.d_z,
        #                                  self.config.d_C_1, self.n_vs,self.p_target)


        # --- optimizer ---
        self.global_step = tf.Variable(0, name="global_step")

        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.1,
                staircase=True,
                name='decaying_learning_rate'
            )

        self.summary_op = tf.summary.merge_all()
        # self.summary_writer = tf.summary.FileWriter('./graph',tf.get_default_graph())
        self.saver = tf.train.Saver()
        # self.checkpoint_secs = 10  # 5 min

        self.train_lv = self.optimize_adam(self.model.lv_loss,
                                              train_vars=self.model.lv_train, lr=1e-2)
        self.train_mu = self.optimize_adam(self.model.mu_loss,
                                              train_vars=self.model.mu_train, lr=1e-2)
        self.train_Sigma = self.optimize_adam(self.model.Sigma_loss,
                                              train_vars=self.model.Sigma_train, lr=1e-2)
        self.train_weight = self.optimize_adam(self.model.weight_loss,
                                              train_vars=self.model.weight_train, lr=1e-5)
        tf.global_variables_initializer().run()

        # init_Prcs=session.run(self.model.q_approx.Prcs)
        # sio.savemat('./train_data_HLS/Prcs_init.mat', {'Prcs_init': init_Prcs})
        #
        # v_g=tf.gradients(self.model.lv_loss,self.model.v)
        # v_g_val=self.session.run(v_g)
        # print(v_g_val)



    def train(self):

        # training q
        it=0
        MIN_KL=1e10
        # count_bad_kl=0
        while it<=self.config.MAXIT:
            start=time.time()
            kl_loss=0
            try:
                for n_updates in range(1, 1 + self.config.max_steps):
                    step, summary, loss = self.run_single_step(self.model.lv_loss,self.train_lv)
                    kl_loss=loss
            except:
                break
            for n_updates in range(1, 1 + self.config.max_steps):
                step, summary, loss = self.run_single_step(self.model.mu_loss,self.train_mu)
                kl_loss=loss
            # Prcs_before = self.session.run(self.model.q_approx.Prcs)
            # sio.savemat('./train_data_HLS/Prcs_before.mat', {'Prcs_before': Prcs_before})
            # v_before = self.session.run(self.model.v)
            # sio.savemat('./train_data_HLS/v_before.mat', {'v_before': v_before})
            # break
            try:
                for n_updates in range(1, 1 + self.config.max_steps):
                    step, summary, loss = self.run_single_step(self.model.Sigma_loss,self.train_Sigma)
                    kl_loss=loss
            except:
                break
            for n_updates in range(1, 1 + self.config.max_steps):
                step, summary, loss = self.run_single_step(self.model.weight_loss, self.train_weight)
            finish=time.time()
            print('iteration time: ', str(finish-start))
            # if it>200 and kl_loss>0.0:
            #     count_bad_kl+=1
            # if count_bad_kl>50:
            #     break


            if it>=self.config.start_update_q:
                self.model.update_q_v()
                if it>=self.config.start_MC_q and it%2==0:
                    #q_var_before = self.session.run(self.model.q_v)
                    #sio.savemat('./train_data_HLS/q_before.mat', {'q_before': q_var_before})
                    sample, kl_arr_val, q_sample_before = self.session.run(self.model.MH_sample())
                    K_b=q_sample_before.shape[1]
                    q_sample_shuffle=np.zeros([1,K_b,self.config.K_r],dtype=np.float32)
                    for j_b in range(K_b):
                        q_sample_shuffle[0,j_b,:]=np.random.permutation(q_sample_before[0,j_b,:])

                    # # print(sample)
                    self.session.run(self.model.MH_sample_update(sample, kl_arr_val, q_sample_shuffle,it))
                    #q_var_after = self.session.run(self.model.q_v)
                    #sio.savemat('./train_data_HLS/q_after.mat', {'q_after': q_var_after})
                # break
                if kl_loss<MIN_KL:
                    MIN_KL = kl_loss
                    weight_hat = self.session.run(self.model.q_approx.weight)
                    sio.savemat('./train_data_HLS/weight_hat_' + str(self.trial_i) + '_min.mat', {'weight_hat': weight_hat})
                    mean_hat = self.session.run(self.model.q_approx.mean)
                    sio.savemat('./train_data_HLS/mu_hat_' + str(self.trial_i) + '_min.mat', {'mean_hat': mean_hat})
                    Prcs_hat = self.session.run(self.model.q_approx.Prcs)
                    sio.savemat('./train_data_HLS/Prcs_hat_' + str(self.trial_i) + '_min.mat', {'Prcs_hat': Prcs_hat})
                    v_var = self.session.run(self.model.v)
                    ps_v_var = self.session.run(self.model.ps_v)
                    H_w_var = self.session.run(self.model.H_w)
                    H_y_var = self.session.run(self.model.H_y)
                    H_z_var = self.session.run(self.model.H_z)
                    b_1_var = self.session.run(self.model.b_1)
                    b_out_var = self.session.run(self.model.b_out)
                    C_var = self.session.run(self.model.C)
                    beta_var = self.session.run(self.model.beta)
                    mu_1_var = self.session.run(self.model.mu_1)
                    mu_out_var = self.session.run(self.model.mu_out)
                    xi_var = self.session.run(self.model.xi)
                    q_var = self.session.run(self.model.q_v)
                    sio.savemat('./train_data_HLS/PML_para_' + str(self.trial_i) + '_min.mat',
                                {'v': v_var, 'ps_v': ps_v_var, 'H_w': H_w_var, 'H_y': H_y_var,
                                 'H_z': H_z_var, 'b_1': b_1_var, 'b_out': b_out_var, 'C': C_var, 'beta': beta_var,
                                 'mu_out': mu_out_var, 'mu_1': mu_1_var, 'xi': xi_var, 'q': q_var,
                                 'config': self.config})

            weight_hat = self.session.run(self.model.q_approx.weight)
            sio.savemat('./train_data_HLS/weight_hat.mat', {'weight_hat': weight_hat})
            mean_hat = self.session.run(self.model.q_approx.mean)
            sio.savemat('./train_data_HLS/mu_hat.mat', {'mean_hat': mean_hat})
            Prcs_hat = self.session.run(self.model.q_approx.Prcs)
            sio.savemat('./train_data_HLS/Prcs_hat.mat', {'Prcs_hat': Prcs_hat})
            q_var = self.session.run(self.model.q_v)
            sio.savemat('./train_data_HLS/q_var.mat', {'q': q_var})

            print('iteration: ', str(it))
            print('loss: ', kl_loss / self.config.n_gmm)
            it += 1
        weight_hat = self.session.run(self.model.q_approx.weight)
        sio.savemat('./train_data_HLS/weight_hat_'+str(self.trial_i)+'.mat', {'weight_hat': weight_hat})
        mean_hat = self.session.run(self.model.q_approx.mean)
        sio.savemat('./train_data_HLS/mu_hat_'+str(self.trial_i)+'.mat', {'mean_hat': mean_hat})
        Prcs_hat = self.session.run(self.model.q_approx.Prcs)
        sio.savemat('./train_data_HLS/Prcs_hat_'+str(self.trial_i)+'.mat', {'Prcs_hat': Prcs_hat})
        v_var = self.session.run(self.model.v)
        ps_v_var=self.session.run(self.model.ps_v)
        H_w_var = self.session.run(self.model.H_w)
        H_y_var = self.session.run(self.model.H_y)
        H_z_var = self.session.run(self.model.H_z)
        b_1_var=self.session.run(self.model.b_1)
        b_out_var = self.session.run(self.model.b_out)
        C_var = self.session.run(self.model.C)
        beta_var = self.session.run(self.model.beta)
        mu_1_var = self.session.run(self.model.mu_1)
        mu_out_var = self.session.run(self.model.mu_out)
        xi_var = self.session.run(self.model.xi)
        q_var=self.session.run(self.model.q_v)
        sio.savemat('./train_data_HLS/PML_para_'+str(self.trial_i)+'.mat', {'v': v_var, 'ps_v':ps_v_var ,'H_w':H_w_var, 'H_y':H_y_var,
                                                                            'H_z':H_z_var, 'b_1':b_1_var,'b_out':b_out_var, 'C':C_var,'beta':beta_var,
                                                                            'mu_out':mu_out_var,'mu_1':mu_1_var, 'xi':xi_var, 'q':q_var,'config':self.config,'final_step':it})

        # v_var = self.session.run(self.model.v)
        # H_w_var = self.session.run(self.model.H_w)
        # H_y_var = self.session.run(self.model.H_y)
        # H_z_var = self.session.run(self.model.H_z)
        # b_var=self.session.run(self.model.b)
        # C_b_1_var = self.session.run(self.model.C_b_1)
        # C_1_var = self.session.run(self.model.C_1)
        # C_out_b_var = self.session.run(self.model.C_out_b)
        # C_out_var = self.session.run(self.model.C_out)
        # mu_var = self.session.run(self.model.mu)
        # xi_var = self.session.run(self.model.xi)
        # q_var=self.session.run(self.model.q_v)
        # sio.savemat('./train_data_HLS/PML_para_'+str(self.trial_i)+'.mat', {'v': v_var, 'H_w':H_w_var, 'H_y':H_y_var,
        #                                                                     'H_z':H_z_var, 'b':b_var, 'C_1':C_1_var,'C_b_1':C_b_1_var,
        #                                                                     'C_out':C_out_var, 'C_out_b':C_out_b_var,'mu':mu_var, 'xi':xi_var, 'q':q_var,
        #                                                                     'config':self.config})




    def run_single_step(self,min_loss,train_model):

        fetch = [self.global_step, self.summary_op, min_loss, train_model]
        fetch_values = self.session.run(fetch)
        [step, summary, loss] = fetch_values[:3]
        return step, summary, loss


def main():
    import argparse

    filename='Synthetic_Data/train_and_test_primal/gmm_train'
    # filename='Flow_Cytometry/input_gmm'

    try:
        gmm_data = sio.loadmat(filename+'.mat')

        mean_p = np.float32(gmm_data['mu_data'])
        print(mean_p.shape)

        Sigma_p = np.float32(gmm_data['Sigma_data'])
        print(Sigma_p.shape)

        weight_p = np.float32(gmm_data['weight_data'])
        print(weight_p.shape)

        try:
            KL_data=sio.loadmat(filename+'_KL_M.mat')
            KL_mat_norm=KL_data['KL_M_norm_py']
        except:

            #diagonal covariance
            [dim, K_b, n_gmm] = mean_p.shape
            Prcs_data = np.zeros((dim, K_b, n_gmm), dtype=np.float32)
            Sigma_data = np.zeros((dim, K_b, n_gmm), dtype=np.float32)
            for i in range(n_gmm):
                for j in range(K_b):
                    Sigma_data[:, j, i] = np.diag(Sigma_p[:, :, j, i])
                    Prcs_data[:, j, i] = 1 / (Sigma_data[:, j, i])
            KL_mat = np.zeros((n_gmm, n_gmm), dtype=np.float32)
            for i in range(n_gmm):
                for j in range(i, n_gmm):
                    KL_mat[i, j] = cal_KL_symm_diag(mean_p[:, :, i], Sigma_data[:, :, i], Prcs_data[:, :, i],
                                                    weight_p[:, i], mean_p[:, :, j], Prcs_data[:, :, j],
                                                    weight_p[:, j]) + \
                                   cal_KL_symm_diag(mean_p[:, :, j], Sigma_data[:, :, j], Prcs_data[:, :, j],
                                                    weight_p[:, j], mean_p[:, :, i], Prcs_data[:, :, i], weight_p[:, i])
                print('finish ' + str(i))

            # full covariance
            # [dim, K_b, n_gmm] = mean_p.shape
            # Prcs_data = np.zeros((dim, dim, K_b, n_gmm), dtype=np.float32)
            # for i in range(n_gmm):
            #     for j in range(K_b):
            #         Prcs_data[:, :, j, i] = np.linalg.inv(Sigma_p[:, :, j, i])
            # KL_mat = np.zeros((n_gmm, n_gmm), dtype=np.float32)
            # for i in range(n_gmm):
            #     for j in range(i, n_gmm):
            #         KL_mat[i, j] = cal_KL_symm(mean_p[:, :, i], Sigma_p[:, :, :, i], Prcs_data[:, :, :, i],
            #                                    weight_p[:, i], mean_p[:, :, j], Prcs_data[:, :, :, j], weight_p[:, j]) + \
            #                        cal_KL_symm(mean_p[:, :, j], Sigma_p[:, :, :, j], Prcs_data[:, :, :, j],
            #                                    weight_p[:, j], mean_p[:, :, i], Prcs_data[:, :, :, i], weight_p[:, i])
            #     print('finish ' + str(i))

            KL_mat_sym = KL_mat + np.transpose(KL_mat)
            # sio.savemat('eye_KL_mat.mat',{'KL_M_py':KL_mat_sym})
            KL_mat_norm = KL_mat_sym/np.max(KL_mat_sym)
            sio.savemat(filename + '_KL_M.mat', {'KL_M_norm_py': KL_mat_norm})

    except:
        gmm_data = h5py.File(filename+'.mat')

        mean_p = np.transpose(np.float32(gmm_data['mu_data']), (2, 1, 0))
        print(mean_p.shape)

        Sigma_p = np.transpose(np.float32(gmm_data['Sigma_data']), (2, 3, 1, 0))
        print(Sigma_p.shape)

        weight_p = np.transpose(np.float32(gmm_data['weight_data']), (1, 0))
        print(weight_p.shape)


        try:
            KL_data=sio.loadmat(filename+'_KL_M.mat')
            KL_mat_norm=KL_data['KL_M_norm_py']
        except:
            [dim, K_b, n_gmm] = mean_p.shape
            Prcs_data = np.zeros((dim, K_b, n_gmm), dtype=np.float32)
            Sigma_data = np.zeros((dim, K_b, n_gmm), dtype=np.float32)
            for i in range(n_gmm):
                for j in range(K_b):
                    Sigma_data[:, j, i] = np.diag(Sigma_p[:, :, j, i])
                    Prcs_data[:, j, i] = 1 / (Sigma_data[:, j, i])
            KL_mat = np.zeros((n_gmm, n_gmm), dtype=np.float32)
            for i in range(n_gmm):
                for j in range(i, n_gmm):
                    KL_mat[i, j] = cal_KL_symm_diag(mean_p[:, :, i], Sigma_data[:, :, i], Prcs_data[:, :, i],
                                                    weight_p[:, i], mean_p[:, :, j], Prcs_data[:, :, j],
                                                    weight_p[:, j]) + \
                                   cal_KL_symm_diag(mean_p[:, :, j], Sigma_data[:, :, j], Prcs_data[:, :, j],
                                                    weight_p[:, j], mean_p[:, :, i], Prcs_data[:, :, i], weight_p[:, i])
                print('finish ' + str(i))
            KL_mat_sym = KL_mat + np.transpose(KL_mat)
            KL_mat_norm = KL_mat_sym/np.max(KL_mat_sym)
            sio.savemat(filename + '_KL_M.mat', {'KL_M_norm_py': KL_mat_norm})






    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=400, required=False) # iterations in updating Gaussian parameters
    parser.add_argument('--MAXIT', type=int, default=3000, required=False)  # max EM iterations
    parser.add_argument('--learning_rate', type=float, default=5e-3, required=False)  # learning rate
    parser.add_argument('--lr_weight_decay', action='store_true', default=True)  # if use learning rate decay
    parser.add_argument('--K_r', type=int, default=9, required=False)  # number of reconstructed components
    parser.add_argument('--d_v', type=int, default=3, required=False)  # dimension of hierarchical latent variables
    parser.add_argument('--d_w', type=int, default=2, required=False)  # dimension of w
    parser.add_argument('--d_y', type=int, default=2, required=False)  # dimension of y
    parser.add_argument('--d_z', type=int, default=2, required=False)  # dimension of z
    parser.add_argument('--d_C_1', type=int, default=20, required=False) # dimension of the hidden layer in Covariance generation neural network, when use Model_HLS_diag_cov
    parser.add_argument('--d_mu_1', type=int, default=4, required=False) # dimension of the hidden layer in mean generation neural network
    parser.add_argument('--reg_v', type=float, default=0.00, required=False) # coefficient of L2 norm regularizer for v
    parser.add_argument('--reg_wyz', type=float, default=0.00, required=False) # coefficient of L2 norm regularizer for w,y,z
    parser.add_argument('--reg_kl', type=float, default=0.01, required=False) # coefficient of KL regularizer
    parser.add_argument('--n_ps_v',type=int,default=20,required=False) # number of pseudo-points in the kernel approximation
    parser.add_argument('--lamda_kern',type=float, default=0.00001,required=False) # lambda in the kernel inverse
    parser.add_argument('--gamma_width',type=float,default=1.0,required=False) # width in the Gaussian kernel
    parser.add_argument('--gamma_scale',type=float,default=1.0,required=False) # scale in the Gaussian kernel
    parser.add_argument('--poly_scale',type=float,default=1.0,required=False) # scale in the polynomial kernel
    parser.add_argument('--T_0', type=float, default=1.0001, required=False) # initial temperature
    parser.add_argument('--T_coeff', type=float, default=0.0008, required=False) # temperature parameters in updating q
    parser.add_argument('--start_update_q', type=int, default=150, required=False) # the step to start to update q
    parser.add_argument('--start_MC_q', type=int, default=250, required=False) # the step to start to MH sampling for q
    parser.add_argument('--n_gmm', type=int, default=mean_p.shape[2], required=False) # number of input GMMs
    parser.add_argument('--n_vs',type=float,default=2.0,required=False) # number of virtual samples
    config = parser.parse_args()


    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
    )
    # repeat 10 times
    for i in range(10):
        with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
            # with tf.device('/device:GPU:0'):
            with tf.device('/cpu:0'):
                trainer = Trainer(config, sess, mean_p, Sigma_p, weight_p,KL_mat_norm, i)
                print('finish initialization')
                trainer.train()
if __name__ == '__main__':
    main()
