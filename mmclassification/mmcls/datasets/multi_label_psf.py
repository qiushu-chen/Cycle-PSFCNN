# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:09:54 2022

@author: gywbc
"""
import numpy as np
from typing import List
from .builder import DATASETS
from .base_dataset import BaseDataset
from mmcls.core import average_performance, mAP

@DATASETS.register_module()
class MultiLabelPSFDataset(BaseDataset):
    def load_annotations(self):
        assert isinstance(self.ann_file, str) 
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            # databuf = 0.5*np.pi
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                gt_label_split = gt_label.split(',')
                gt_label_array = np.zeros(len(gt_label_split))
                for i in range(len(gt_label_split)):
                    gt_label_array[i] = np.array(gt_label_split[i], dtype=np.float32)
                    # if i==1:
                    #     gt_label_array[i] = gt_label_array[i] + 2.0
                    # elif i==3:
                    #     gt_label_array[i] = gt_label_array[i] + databuf
                #info['gt_label'] = np.array(gt_label, dtype=np.int64)
                info['gt_label'] = gt_label_array
                data_infos.append(info)
            return data_infos
        
    def evaluate(self,
                 results,
                 metric='mAP',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict: evaluation results
        """
        if metric_options is None or metric_options == {}:
            metric_options = {'thr': 0.5}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if len(set(metrics) - {'mAP'}) != 0:
            performance_keys = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
            performance_values = average_performance(results, gt_labels,
                                                     **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v

        return eval_results