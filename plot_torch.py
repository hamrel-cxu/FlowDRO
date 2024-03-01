import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_attack_errs(attack_mtd):
    def get_mean_ave(curr_list):
        curr_list = np.array(curr_list)
        avg = curr_list.mean(axis=0)
        std = curr_list.std(axis=0)
        return avg, std
    wrm_torch_ell2_ls, wrm_torch_ellinf_ls = [], []
    frm_torch_ell2_ls, frm_torch_ellinf_ls = [], []
    for run in run_ls:
        wrm_torch = np.load(os.path.join(dir_name2, f'wrm_{attack_mtd}_step{FRM_steps}_run{run}.npy'), allow_pickle=True).item()
        frm_torch = np.load(os.path.join(dir_name2, f'frm_{attack_mtd}_step{FRM_steps}_run{run}.npy'), allow_pickle=True).item()
        wrm_torch_ell2_ls.append(wrm_torch['ell_2'])
        wrm_torch_ellinf_ls.append(wrm_torch['ell_inf'])
        frm_torch_ell2_ls.append(frm_torch['ell_2'])
        frm_torch_ellinf_ls.append(frm_torch['ell_inf'])
    wrm_torch_ell2, wrm_torch_ell2_std = get_mean_ave(wrm_torch_ell2_ls)
    wrm_torch_ellinf, wrm_torch_ellinf_std = get_mean_ave(wrm_torch_ellinf_ls)
    frm_torch_ell2, frm_torch_ell2_std = get_mean_ave(frm_torch_ell2_ls)
    frm_torch_ellinf, frm_torch_ellinf_std = get_mean_ave(frm_torch_ellinf_ls)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fontsize = 24
    msize = 5
    ax[0].plot(fracs_ell2, wrm_torch_ell2, '-o', color='red', label='WRM', markersize=msize)
    ax[0].plot(fracs_ell2, frm_torch_ell2, '-o', color='blue', label='FRM', markersize=msize)
    ax[0].fill_between(fracs_ell2, wrm_torch_ell2 - wrm_torch_ell2_std, wrm_torch_ell2 + wrm_torch_ell2_std, alpha=0.2, color='red')
    ax[0].fill_between(fracs_ell2, frm_torch_ell2 - frm_torch_ell2_std, frm_torch_ell2 + frm_torch_ell2_std, alpha=0.2, color='blue')
    ax[0].set_ylabel('Error', fontsize=fontsize)
    ax[0].set_xlabel(r'Attack budget $\epsilon_{\rm{adv}}/C_2$', fontsize=fontsize)
    ax[1].plot(fracs_ellinf, wrm_torch_ellinf, '-o', color='red', label='WRM', markersize=msize)
    ax[1].plot(fracs_ellinf, frm_torch_ellinf, '-o', color='blue', label='FRM', markersize=msize)
    ax[1].fill_between(fracs_ellinf, wrm_torch_ellinf - wrm_torch_ellinf_std, wrm_torch_ellinf + wrm_torch_ellinf_std, alpha=0.2, color='red')
    ax[1].fill_between(fracs_ellinf, frm_torch_ellinf - frm_torch_ellinf_std, frm_torch_ellinf + frm_torch_ellinf_std, alpha=0.2, color='blue')
    ax[1].set_ylabel('Error', fontsize=fontsize)
    ax[1].set_xlabel(r'Attack budget $\epsilon_{\rm{adv}}/C_{\infty}$', fontsize=fontsize)
    for a in ax.flatten():
        a.legend(loc = 'upper left', fontsize=fontsize)
        a.tick_params(axis='both', which='major', labelsize=16)
    fig.suptitle(f'Classification error under {attack_mtd} attack', fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_name2, f'defend_{attack_mtd}_step{FRM_steps}_avg.png'), bbox_inches='tight', pad_inches=0.5)

parser = argparse.ArgumentParser()
parser.add_argument('--full', type=int, default=0, choices=[0, 1])
args = parser.parse_args()
two_digits = [int(d) for d in '6-7'.split('-')]
FRM_steps = 3
WRM_steps = 15 if args.full == 0 else 100
if __name__ == '__main__':
    if args.full == 1:
        dir_name = os.path.join('models', 'mnist_full')
        dir_name2 = os.path.join('results', 'mnist_full')
    else:
        dir_name = os.path.join('models', f'mnist_binary')
        dir_name2 = os.path.join('results', f'mnist_binary')
    suffix = f'_step{FRM_steps}'
    run_ls = [0, 1, 2]
    ############## Plot the attack errors
    fracs_ell2 = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    fracs_ellinf = [0, 0.1, 0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]
    plot_attack_errs('PGD')