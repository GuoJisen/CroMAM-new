import random
import pandas as pd
import numpy as np
import glob
import os
import argparse
import logging
from sklearn.model_selection import StratifiedKFold
from utils.helper import get_filename_extensions

parser = argparse.ArgumentParser(description='Create meta information')
parser.add_argument('--ffpe-only',
                    action='store_true', default=False,
                    help='keep only ffpe slides')
parser.add_argument('--cancer',
                    type=str, default='LGG',
                    help='Cancer type')
parser.add_argument('--magnification',
                    type=str, default=5,
                    help='magnification level')
parser.add_argument('--patch-size',
                    type=int, default=224,
                    help='size of the extracted patch')
parser.add_argument('--random-seed',
                    type=int, default=88,
                    help='random seed for generating the Nested-CV splits')
# K FOLDS SPLIT
parser.add_argument('--outer-fold',
                    type=int, default=5,
                    help='number of outer folds for the Nested-CV splits')
parser.add_argument('--inner-fold',
                    type=int, default=9,
                    help='number of inner folds for the Nested-CV splits')
parser.add_argument('--root',
                    type=str, default='/home/dell/projects/gjs_workstation/WSI-PLP/CroMAM',
                    help='root directory')
parser.add_argument('--patch_root',
                    type=str, default='/media/dell/data/DATA/pj_data/PLP',
                    help='patch root directory')

parser.add_argument('--stratify',
                    type=str, default='idh',
                    help='when spliting the datasets, stratify on which variable')

args = parser.parse_args()
np.random.seed(args.random_seed)
random.seed(args.random_seed)

magnification = list(args.magnification.split(','))
EXT_DATA, EXT_EXPERIMENT, EXT_SPLIT = get_filename_extensions(args)
logging_file = '%s/logs/meta_log_%s_%s.csv' % (args.root, EXT_DATA[0], EXT_EXPERIMENT)
handlers = [logging.FileHandler(logging_file, mode='w'), logging.StreamHandler()]
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=handlers)

for arg, value in sorted(vars(args).items()):
    logging.info("Argument %s: %s" % (arg, value))


def get_patch_meta(patch_dir, ext, root_dir):
    if os.path.exists('%s/dataset/patches_meta_raw_%s.pickle' % (root_dir, ext)):
        df_cmb = pd.read_pickle('%s/dataset/patches_meta_raw_%s.pickle' % (root_dir, ext))
    else:
        patch_files = glob.glob('%s/*/*/*/*.jpg' % patch_dir)
        patch_files = sorted(patch_files, key=lambda name: name.split('/')[-3])
        logging.info("Number of patch files (no deal): %s" % len(patch_files))
        if len(patch_files) == 0:
            return 0
        df_cmb = pd.DataFrame(columns=['file'])
        df_cmb['file'] = patch_files
        df_cmb['file_original'] = df_cmb.file.apply(lambda x: x.split('/')[-3])
        df_cmb['submitter_id'] = df_cmb.file_original.apply(lambda x: x[:12])

    df_cmb_meta = df_cmb.drop_duplicates('file_original').copy()  # 删去该列重复的数据
    df_cmb_meta['slide_type'] = df_cmb_meta.file_original.apply(lambda x: x.split('-')[3])

    df_cmb_meta['ffpe_slide'] = 0
    df_cmb_meta.loc[df_cmb_meta.slide_type.str.contains('01Z|02Z|DX'), 'ffpe_slide'] = 1

    df_cmb = df_cmb.merge(df_cmb_meta[['file_original', 'slide_type', 'ffpe_slide']], on='file_original', how='inner')

    if args.ffpe_only:
        df_cmb = df_cmb.loc[df_cmb.ffpe_slide == 1].reset_index(drop=True)
    logging.info("Number of final patch files finally: %s" % len(df_cmb.submitter_id))
    logging.info("Number of patients in the final dataset: %s" % len(df_cmb.submitter_id.unique()))

    return df_cmb


def classification_data_perpare(df):
    df[args.stratify] = 0
    if args.stratify == 'idh':
        df.loc[df.idh_status == 'm', args.stratify] = [1 if x == 'm' else 0 for x in
                                                       df[df.idh_status == 'm'].idh_status.to_list()]
        df = df.loc[~df.idh_status.isna()].copy().reset_index(drop=True)
    elif args.stratify == '1p19q':
        df.loc[df.pq_status == 'm', args.stratify] = [1 if x == 'm' else 0 for x in
                                                      df[df.pq_status == 'm'].pq_status.to_list()]
        df = df.loc[~df.pq_status.isna()].copy().reset_index(drop=True)
    elif args.stratify == 'sur':
        df.loc[df.sur_status == 1, args.stratify] = [1 if x == 1 else 0 for x in
                                                     df[df.sur_status == 1].sur_status.to_list()]
        df = df.loc[~df.sur_status.isna()].copy().reset_index(drop=True)
    else:
        pass
    logging.info('number of participants after excluding missing time %s' % df.shape[0])
    return df


def random_split_by_id_compare(df_cmb, df_meta, root_dir='../'):
    df_split = pd.DataFrame()
    p_num = pd.DataFrame(df_cmb.groupby(['submitter_id']).size(), columns=['num_patches']).reset_index(inplace=False)
    vars_to_keep = ['submitter_id', 'stratify_var']
    if args.stratify:
        df_meta['stratify_var'] = df_meta[args.stratify]
    else:
        df_meta['stratify_var'] = np.random.randint(0, 2, df_meta.shape[0])

    df = df_cmb[['submitter_id']].merge(df_meta[vars_to_keep], on='submitter_id', how='inner')
    df = df.dropna()

    df_id = df.drop_duplicates('submitter_id').reset_index(drop=True).copy()[vars_to_keep]
    logging.info("Total number of patients: %s" % df_id.shape[0])

    df_id['split'] = 0
    df_id.reset_index(drop=True, inplace=True)

    kf_outer = StratifiedKFold(args.outer_fold, random_state=args.random_seed, shuffle=True)

    for i, (tr_index, val_index) in enumerate(kf_outer.split(df_id, df_id['stratify_var'])):

        logging.info("-" * 40)
        df_train = df_id.loc[df_id.index.isin(tr_index)].reset_index(drop=True)
        df_val = df_id.loc[df_id.index.isin(val_index)].reset_index(drop=True)
        logging.info("Working on outer split %s .... Train: %s; Val: %s" % (i, df_train.shape[0], df_val.shape[0]))

        dt = df_train.merge(p_num, on='submitter_id', how="left")
        dv = df_val.merge(p_num, on='submitter_id', how="left")
        df_split[[f"fold{i}_t", f"t_nums{i}", f"t_lable{i}"]] = dt[["submitter_id", "num_patches", 'stratify_var']]
        df_split[[f"fold{i}_v", f"v_nums{i}", f"v_lable{i}"]] = dv[["submitter_id", "num_patches", 'stratify_var']]

        df_train[['submitter_id', 'split']]. \
            merge(df_meta, on='submitter_id', how='inner'). \
            to_pickle(f'{root_dir}/dataset/{args.cancer}_{args.stratify}_meta_train_x{magnification[0]}_{i}.pickle')
        df_val[['submitter_id', 'split']]. \
            merge(df_meta, on='submitter_id', how='inner'). \
            to_pickle(f'{root_dir}/dataset/{args.cancer}_{args.stratify}_meta_val_x{magnification[0]}_{i}.pickle')
    df_split.to_csv(
        f"{args.root}/logs/{args.cancer}_{args.stratify}_data_split_{args.magnification}.csv")


if __name__ == '__main__':
    # process meta information / meta file path
    fname_meta = args.root + '/dataset/meta_files/meta_clinical_%s_%s.csv' % (args.cancer, args.stratify)
    if os.path.isfile(fname_meta):
        # read csv
        df_meta = pd.read_csv(fname_meta)
        df_meta = classification_data_perpare(df_meta)
        print(df_meta.head())
    else:
        pass
    logging.info(df_meta.describe())

    df_cmbs = []
    df_cmb = 0
    for i in range(len(EXT_DATA)):
        try:
            patch_dir = args.patch_root + '/%s/%s_%s' % (args.cancer, magnification[i], args.patch_size)  # patches path
            # process patch information
            patch_meta_file = '%s/dataset/patches_meta_%s.pickle' % (args.root, EXT_DATA[i])  # patch_meta_file path
            if os.path.exists(patch_meta_file):
                logging.info("patch meta file %s already exists!" % patch_meta_file)
                logging.info("patch num : %s " % len(pd.read_pickle(patch_meta_file)))
                df_cmbs.append(pd.read_pickle(patch_meta_file))
            else:
                df_cmbs.append(get_patch_meta(patch_dir, EXT_DATA[i], args.root))
                df_cmbs[i].to_pickle(patch_meta_file)
            p_nums = pd.DataFrame(df_cmbs[i].groupby(['submitter_id']).size(), columns=['num_patches'])
            p_nums.to_csv(
                f"{args.root}/logs/{args.cancer}_{args.stratify}_patch_num_{magnification[i]}.csv")
        except Exception as e:
            print(e)
    if len(EXT_DATA) != 1:
        df1_nums = df_cmbs[0].drop_duplicates('submitter_id').reset_index(drop=True)
        df2_nums = df_cmbs[1].drop_duplicates('submitter_id').reset_index(drop=True)
        if df1_nums.shape[0] > df2_nums.shape[0]:
            df_cmb = df_cmbs[1]
        elif df1_nums.shape[0] <= df2_nums.shape[0]:
            df_cmb = df_cmbs[0]
    else:
        df_cmb = df_cmbs[0]
    random_split_by_id_compare(df_cmb, df_meta, args.root)
