#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: count.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from src.lstm import LSTMcell
import src.assign as assign

def count_0_in_seq(input_seq, count_type):
    """ count number of digit '0' in input_seq

    Args:
        input_seq (list): input sequence encoded as one hot
            vectors with shape [num_digits, 10].
        count_type (str): type of task for counting. 
            'task1': Count number of all the '0' in the sequence.
            'task2': Count number of '0' after the first '2' in the sequence.
            'task3': Count number of '0' after '2' but erase by '3'.

    Return:
        counts (int)
    """
    cell = LSTMcell(in_dim=10, out_dim=1)
    if count_type == 'task1':
        # Count number of all the '0' in the sequence.


        # assign parameters
        assign.assign_weight_count_all_0_case_1(cell, in_dim=10, out_dim=1)

        # initial the first state
        prev_state = [0.]
        # read input sequence one by one to count the digits
        for idx, d in enumerate(input_seq):
            prev_state = cell.run_step([d], prev_state=prev_state)

        count_num = int(np.squeeze(prev_state))
        return count_num

    if count_type == 'task2':
        # Count number of '0' after the first '2' in the sequence.

        cell_2 = LSTMcell(in_dim=10, out_dim=1)
        assign.assign_weight_count_all_0_case_2(cell_2, in_dim=10, out_dim=1, idx =2, forget =100)
        assign.assign_weight_count_all_0_case_2(cell, in_dim=10, out_dim =1, idx =0, forget = 0)

        prev_state_2 = [0.]
        prev_state = [0.]
        for idx, d in enumerate(input_seq):
            prev_state = cell.run_step([d], prev_state = prev_state)
            prev_state_2 = cell_2.run_step([d], prev_state = prev_state_2)

            if(int(np.squeeze(prev_state_2))):

                #update to start count
                assign.assign_weight_count_all_0_case_2(cell, in_dim = 10, out_dim =1, idx =0, forget =100.)
                assign.assign_weight_count_all_0_case_2(cell_2, in_dim = 10, out_dim =1, idx =2, forget =0.)



        count_num = int(np.squeeze(prev_state))
        return count_num

    if count_type == 'task3':

        prev_state = [0.]
        prev_state_2 = [0.]
        prev_state_3 = [0.]
        #create 2 cells for to find 2 another for 3

        cell_2 = LSTMcell(in_dim=10, out_dim=1)
        cell_3 = LSTMcell(in_dim=10, out_dim=1)
        assign.assign_weight_count_all_0_case_2(cell, in_dim=10, out_dim =1,idx =0, forget =0)
        assign.assign_weight_count_all_0_case_2(cell_2, in_dim=10, out_dim =1, idx =2)
        assign.assign_weight_count_all_0_case_2(cell_3, in_dim=10, out_dim =1, idx =3, forget =0)
        for idx, d in enumerate(input_seq):
            prev_state = cell.run_step([d], prev_state = prev_state)
            prev_state_2 = cell_2.run_step([d], prev_state=prev_state_2)
            prev_state_3 = cell_3.run_step([d], prev_state=prev_state_3)
            if(int(np.squeeze(prev_state_2))):
                #first 2 found

                #update weight for 2
                assign.assign_weight_count_all_0_case_2(cell_2, in_dim=10, out_dim=1, idx=2, forget=0)
                #update nn to find 3
                assign.assign_weight_count_all_0_case_2(cell_3, in_dim=10, out_dim=1, idx=3, forget=100)

                #update weight of 0  start count
                assign.assign_weight_count_all_0_case_2(cell, in_dim = 10, out_dim =1, idx =0, forget =100.)

            elif(int(np.squeeze(prev_state_3))):
                #found 3 erase 0 and update weight to find 2
                assign.assign_weight_count_all_0_case_2(cell, in_dim=10, out_dim=1, idx=0, forget=0)
                assign.assign_weight_count_all_0_case_2(cell_3, in_dim=10, out_dim=1, idx=0, forget=0)
                assign.assign_weight_count_all_0_case_2(cell_2, in_dim=10, out_dim=1, idx=2, forget=100)





        # Count number of '0' in the sequence when receive '2', but erase
        # the counting when receive '3', and continue to count '0' from 0
        # until receive another '2'.

        count_num = int(np.squeeze(prev_state))
        return count_num



        