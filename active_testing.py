import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import pickle
import copy
import random
from tools.utils_at import *

random_seed_set = [4519, 9524, 5901]

def LURE_weights_for_risk_estimator(weights, N):
    M = weights.size
    if M < N:
        m = np.arange(1, M+1)
        v = (
            1
            + (N-M)/(N-m) * (
                    1 / ((N-m+1) * weights)
                    - 1
                    )
            )
    else:
        v = 1

    return v

def acquire(expected_loss_inputs, samples_num):
    assert samples_num <= expected_loss_inputs.size
    expected_loss = np.copy(expected_loss_inputs)
    # Log-lik can be negative.
    # Make all values positive.
    if (expected_loss < 0).sum() > 0:
        expected_loss += np.abs(expected_loss.min())
    
    if np.any(np.isnan(expected_loss)):
        logging.warning(
            'Found NaN values in expected loss, replacing with 0.')
        logging.info(f'{expected_loss}')
        expected_loss = np.nan_to_num(expected_loss, nan=0)
    pick_sample_idxs = np.zeros((samples_num), dtype = int)
    idx_array = np.arange(expected_loss.size)
    weights = np.zeros((samples_num), dtype = np.single)
    uniform_clip_val = 0.2
    expected_loss = np.asarray(expected_loss).astype('float64')
    for i in range(samples_num):
        expected_loss /= expected_loss.sum()
        # clip all values less than 10 percent of uniform propability
        expected_loss = np.maximum(uniform_clip_val * 1/expected_loss.size, expected_loss)
        expected_loss /= expected_loss.sum()
        sample = np.random.multinomial(1, expected_loss)
        cur_idx = np.where(sample)[0][0]
        # cur_idx = np.random.randint(expected_loss.size)
        pick_sample_idxs[i] = idx_array[cur_idx]
        weights[i] = expected_loss[cur_idx]
        selected_mask = np.ones((expected_loss.size), dtype=bool)
        selected_mask[cur_idx] = False
        expected_loss = expected_loss[selected_mask]
        idx_array = idx_array[selected_mask]
    return pick_sample_idxs, weights

def active_testing(file_path, true_losses, expected_losses, active_test_type, sample_size_set, display = False, store_idxs = False):
    json_object = {}
    for seed in random_seed_set:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pick_sample_idxs, weights = acquire(expected_losses, sample_size_set[-1])
        if store_idxs:
            np_write(pick_sample_idxs, result_json_path + f"{seed}_at_idxs.npy")
        for sample_size in sample_size_set:
            result = {"active_test_type": active_test_type, "sample_size": sample_size}
            risk_estimator_weights = LURE_weights_for_risk_estimator(weights[:sample_size], expected_losses.size)
            sampled_true_losses = true_losses[pick_sample_idxs[:sample_size]]
            loss_risk = (sampled_true_losses * risk_estimator_weights).mean()
            result["loss"] = loss_risk
            json_object[len(json_object)] = result
        if display:
            print(f"Complete seed : {seed}")
    with open(file_path, "w") as outfile:
        json.dump(json_object, outfile)


def main(args):
    split = args.split
    ck_nums = args.ck_nums
    model_data_type = args.model_data_type
    if model_data_type[-1].isnumeric():
        model_data_origin = "_".join(model_data_type.split("_")[:-1])
    else:
        model_data_origin = model_data_type
    base_path = f"./pro_data/{model_data_origin}/{split}"
    annotation_path = base_path + "/annotation/"
    data_type = args.data_type
    result_json_path = f"./results/{model_data_type}/"
    create_folder_if_not_exists(result_json_path)
    if data_type == "image":
        result_json_path = result_json_path + "image_based_active_testing/"
        true_losses = np_read(base_path + "/image_true_losses.npy")
        sample_size_precentage = np.linspace(0.001, 0.1, 500)
    elif data_type == "region":
        result_json_path = result_json_path + "region_based_active_testing/"
        true_losses = np_read(base_path + "/region_true_losses.npy")
        sample_size_precentage = np.linspace(0.0001, 0.02, 500)
    labels_nums = true_losses.shape[0]
    sample_size_set = (np.array(sample_size_precentage) * labels_nums).astype(int).tolist()
    vit_base_path = "../ViT-pytorch/output/"
    create_folder_if_not_exists(result_json_path)
    

    # random sample
    file_path = result_json_path + "random_sample_3_runs.json"
    json_object = {}
    for seed in random_seed_set:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        perm = np.random.permutation(true_losses.size)
        samples_num = sample_size_set[-1]
        pick_sample_idxs = perm[:samples_num]
        if False:
            np_write(pick_sample_idxs, result_json_path+f"{seed}_rs_idxs.npy")
        for sample_size in sample_size_set:
            result = {"active_test_type": "random sample", "sample_size": sample_size}
            loss_risk = true_losses[pick_sample_idxs[:sample_size]].mean()
            result["loss"] = float(loss_risk)
            json_object[len(json_object)] = result
    write_json_results(json_object, file_path)

    # whole dataset
    file_path = result_json_path + "None.json"
    result = {"active_test_type": "None", "sample_size": true_losses.size}
    result["loss"] = get_whole_data_set_risk_estimator(true_losses)
    json_object = {}
    json_object[0] = result
    write_json_results(json_object, file_path)

    train_steps = np.linspace(10000, 10000*ck_nums, ck_nums, dtype=int)
    for train_step in train_steps:
        val_estimated_loss = np.array(read_json_results(f"{vit_base_path}/ViT_{model_data_type}_{args.loss_range}_{data_type}_losses_{train_step}.json")['losses'])
        file_path = result_json_path + f"ViT_all_runs_{train_step}.json"
        active_testing(file_path, true_losses, val_estimated_loss, "ViT all", sample_size_set, store_idxs=False)

def get_whole_data_set_risk_estimator(true_losses):
    return float(true_losses.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_data_type', type=str, default="UNet_VOC",
                        help="mode X dataset type, current model type: PSPNet, UNet, DeepLab, FCN, SEGNet, dataset: VOC, CITY, COCO, ADE20k")
    parser.add_argument("--data_type", type=str, default="image",
                        help="Region or image.")
    parser.add_argument("--split", default="val", type=str,
                        help="val/train")
    parser.add_argument("--ck_nums", default=10, type=int,
                        help="number of checkpoints")
    parser.add_argument("--loss_range", default="all", type=str,
                        help="loss_range")
    args = parser.parse_args()
    main(args)
    print(f"Complete {args.model_data_type} {args.data_type}")