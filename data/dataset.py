import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from data.table_dataset import SlideDatasetFromTable


def prepare_table_datasets(
        args,
        pickle_file,
        transform,
        mode='train'):
    if isinstance(pickle_file, str):
        _df = pd.read_pickle(pickle_file)
    else:
        _df = pickle_file

    meta_df = {}

    if mode == 'train':
        meta_df[mode] = grouped_sample(  # Group sampling
            _df,
            num_obs=len(_df.submitter_id.unique()) * args.repeats_per_epoch,
            num_patches=args.num_patches,
            patient_var='submitter_id',
            seed=args.random_seed
            )
    elif mode == 'val':
        ids = _df['submitter_id'].unique()
        all_num = int(args.num_val)
        lists = []
        for i in ids:
            num = len(_df[_df.submitter_id == i])
            if num >= all_num:
                ls = _df[_df.submitter_id == i].sample(n=all_num, replace=False, random_state=args.random_seed)
            else:
                ls1 = _df[_df.submitter_id == i].sample(n=num, replace=False, random_state=args.random_seed)
                ls2 = _df[_df.submitter_id == i].sample(n=all_num-num, replace=True, random_state=args.random_seed)
                ls = pd.concat([ls1, ls2]).reset_index(drop=True)
            lists.append(ls)
        meta_df[mode] = pd.concat(lists, axis=0).reset_index(drop=True)

    data = SlideDatasetFromTable(
        data_file=meta_df[mode],
        image_dir='',
        crop_size=args.crop_size,
        outcome=args.outcome,
        transform=transform[mode]
    )

    return data


def unique_shuffle_to_list(x):
    x = x.unique()
    np.random.shuffle(x)
    return x.tolist()


def grouped_sample(data, num_obs=10, num_patches=4, patient_var='submitter_id',
                   patch_var='file', seed=0):
    # step 1: sample patients
    data_meta = data[[patient_var]].drop_duplicates()
    groups = []
    random_seed = 0
    while len(groups) < num_obs:
        random_seed += seed
        groups.extend(
            data_meta.sample(frac=1., random_state=random_seed)[patient_var].tolist())

    # post processing
    groups = groups[:num_obs]

    dfg = pd.DataFrame(groups, columns=[patient_var])
    dfg['queue_order'] = dfg.index
    dfg = dfg.merge(data_meta, on=patient_var).sort_values('queue_order').reset_index(drop=True)

    # step 2: sample patches
    # get the order of occurrence for each patient
    dfg['dummy_count'] = 1
    dfg['within_index'] = dfg.groupby(patient_var).dummy_count.cumsum() - 1  # Repeat x rounds

    ids = data[patient_var].unique()
    all_num = int(num_patches * (dfg.within_index.max() + 1))
    lists = []
    for i in ids:
        num = len(data[data.submitter_id == i])
        if num >= all_num:
            ls = data[data.submitter_id == i].sample(n=all_num, replace=False, random_state=seed)
        else:
            ls1 = data[data.submitter_id == i].sample(n=num, replace=False, random_state=seed)
            ls2 = data[data.submitter_id == i].sample(n=all_num - num, replace=True, random_state=seed)
            ls = pd.concat([ls1, ls2]).reset_index(drop=True)
        lists.append(ls)
    dfp = pd.concat(lists, axis=0).reset_index(drop=True)

    # for each patient, determine merge to which occurrence
    dfp['within_index'] = dfp.groupby(patient_var)[patch_var].transform(
        lambda x: np.arange(x.shape[0]) // num_patches).reset_index(drop=True)

    df_sel = dfg.merge(dfp, on=[patient_var, 'within_index'], how='left')
    return df_sel


def get_multi_dataloader(args, transform, folds):
    """
    to ensure that multiple patches input simultaneously come from the same patient
    """
    magnification = list(args.magnification.split(','))
    if args.mode == 'train':
        # load dataset
        df_patches1 = pd.read_pickle(args.patch_meta + f'_{args.cancer}_{args.outcome}_mag-' + f"{magnification[0]}_size-224.pickle")
        df_patches2 = pd.read_pickle(args.patch_meta + f'_{args.cancer}_{args.outcome}_mag-' + f"{magnification[1]}_size-224.pickle")
        df_patches1['id_patient'] = df_patches1.submitter_id.astype('category').cat.codes
        df_patches2['id_patient'] = df_patches2.submitter_id.astype('category').cat.codes

        df_train1 = pd.read_pickle(args.train_meta + f"{args.cancer}_{args.outcome}_meta_train_x{magnification[0]}_{folds}.pickle")
        df_val1 = pd.read_pickle(args.val_meta + f"{args.cancer}_{args.outcome}_meta_val_x{magnification[0]}_{folds}.pickle")
        df_train2 = pd.read_pickle(args.train_meta + f"{args.cancer}_{args.outcome}_meta_train_x{magnification[1]}_{folds}.pickle")
        df_val2 = pd.read_pickle(args.val_meta + f"{args.cancer}_{args.outcome}_meta_val_x{magnification[1]}_{folds}.pickle")

        cols_to_use = df_patches1.columns.difference(df_train1.columns)
        cols_to_use = cols_to_use.tolist()
        cols_to_use.append('submitter_id')

        df_train1 = df_train1.merge(df_patches1[cols_to_use], on='submitter_id', how='left')
        df_val1 = df_val1.merge(df_patches1[cols_to_use], on='submitter_id', how='left')
        df_train2 = df_train2.merge(df_patches2[cols_to_use], on='submitter_id', how='left')
        df_val2 = df_val2.merge(df_patches2[cols_to_use], on='submitter_id', how='left')

        outcomes = [args.outcome]

        df_train1.dropna(subset=outcomes, inplace=True)
        df_val1.dropna(subset=outcomes, inplace=True)
        df_train2.dropna(subset=outcomes, inplace=True)
        df_val2.dropna(subset=outcomes, inplace=True)

        # prepare_datasets
        train_data1 = prepare_table_datasets(args, df_train1, transform, 'train')
        val_data1 = prepare_table_datasets(args, df_val1, transform, 'val')
        train_data2 = prepare_table_datasets(args, df_train2, transform, 'train')
        val_data2 = prepare_table_datasets(args, df_val2, transform, 'val')

        # create dataloader
        train_loader1 = DataLoader(
            train_data1,
            shuffle=False,
            batch_size=args.batch_size * args.num_patches,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader1 = DataLoader(
            val_data1,
            shuffle=False,
            batch_size=args.vbatch_size * args.num_val,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        train_loader2 = DataLoader(
            train_data2,
            shuffle=False,
            batch_size=args.batch_size * args.num_patches,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader2 = DataLoader(
            val_data2,
            shuffle=False,
            batch_size=args.vbatch_size * args.num_val,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        loader = {'train1': train_loader1, 'val1': val_loader1, 'train2': train_loader2, 'val2': val_loader2}

        return loader
    else:
        print('data loader error check mode')
        pass
