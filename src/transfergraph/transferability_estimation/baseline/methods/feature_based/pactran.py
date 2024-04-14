import numpy as np
import scipy


class PACTran(object):
    def __init__(self, args):
        self.args = args

    def one_hot(self, a):
        b = np.zeros((a.size, a.max() + 1))
        b[np.arange(a.size), a] = 1.
        return b

    def score(self, features_np_all, label_np_all):
        # if lda_factor == 10.:
        #     s2s = [1000., 100.]
        # elif lda_factor == 1.:
        #     s2s = [100., 10.]
        # elif lda_factor == 0.1:
        #     s2s = [10., 1.]

        lda_factor = float(self.args.method.split("-")[1])
        s2_factor = float(self.args.method.split("-")[2])

        """Compute the PAC_Gauss score with diagonal variance."""
        nclasses = label_np_all.max() + 1
        label_np_all = self.one_hot(label_np_all)  # [n, v]

        mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
        features_np_all -= mean_feature  # [n,k]

        bs = features_np_all.shape[0]
        kd = features_np_all.shape[-1] * nclasses
        ldas2 = lda_factor * bs  # * features_np_all.shape[-1]
        dinv = 1. / float(features_np_all.shape[-1])

        # optimizing log lik + log prior
        def pac_loss_fn(theta):
            theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

            w = theta[:features_np_all.shape[-1], :]
            b = theta[features_np_all.shape[-1]:, :]
            logits = np.matmul(features_np_all, w) + b

            log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
            xent = np.sum(
                np.sum(
                    label_np_all * (np.log(label_np_all + 1e-10) - log_qz), axis=-1
                )
            ) / bs
            loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
            return loss

        # gradient of xent + l2
        def pac_grad_fn(theta):
            theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

            w = theta[:features_np_all.shape[-1], :]
            b = theta[features_np_all.shape[-1]:, :]
            logits = np.matmul(features_np_all, w) + b

            grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
            grad_f -= label_np_all
            grad_f /= bs
            grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
            grad_w += w / ldas2

            grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
            grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
            return grad

        # 2nd gradient of theta (elementwise)
        def pac_grad2(theta):
            theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

            w = theta[:features_np_all.shape[-1], :]
            b = theta[features_np_all.shape[-1]:, :]
            logits = np.matmul(features_np_all, w) + b

            prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
            grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
            xx = np.square(features_np_all)  # [n, d]

            grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
            grad2_w += 1. / ldas2
            grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
            grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
            return grad2

        kernel_shape = [features_np_all.shape[-1], nclasses]
        theta = np.random.normal(size=kernel_shape) * 0.03
        theta_1d = np.ravel(
            np.concatenate(
                [theta, np.zeros([1, nclasses])], axis=0
            )
        )

        theta_1d = scipy.optimize.minimize(
            pac_loss_fn, theta_1d, method="L-BFGS-B",
            jac=pac_grad_fn,
            options=dict(maxiter=100), tol=1e-6
        ).x

        pac_opt = pac_loss_fn(theta_1d)

        h = pac_grad2(theta_1d)
        sigma2_inv = np.sum(h) * ldas2 / kd + 1e-10

        s2 = s2_factor * dinv
        pac_gauss = pac_opt + 0.5 * kd / ldas2 * s2 * np.log(
            sigma2_inv
        )

        return pac_gauss
