import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below

        correct = self.predictions == self.actuals
        self.acc = float(Counter(correct)[True])/len(correct)
        predictions = self.predictions
        actuals = self.actuals
        matrix_dict = {}
        for class_n in self.classes_:
            
            tp = np.sum(np.logical_and(predictions==class_n, actuals == class_n))
            tn = np.sum(np.logical_and(predictions!=class_n, actuals != class_n))
            fp = np.sum(np.logical_and(predictions==class_n, actuals != class_n))
            fn = np.sum(np.logical_and(predictions!=class_n, actuals == class_n))
            matrix_dict[class_n] = {"TP":tp, "TN": tn, "FP": fp, "FN": fn}

        self.confusion_matrix = matrix_dict
        
        return


    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()

        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp+fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            if average == "micro":
                
                prec = self.acc
            elif average == 'macro':
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    ratio = 1 / len(self.classes_)
                    prec += prec_label * ratio
            elif average == 'weighted':
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    ratio = Counter(self.actuals)[label] / float(n)
                    prec += prec_label * ratio
            else:
                raise Exception("Invalid average input, please select macro, micro, or weighted")


        return prec

    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()

        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if tp+fn == 0:
                rec = 0
            else:
                rec = float(tp) / (tp + fn)
        else:
            if average == "micro":
                rec = self.acc
            elif average == 'macro':
                rec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        rec_label = 0
                    else:
                        rec_label = float(tp) / (tp + fn)
                    ratio = 1 / len(self.classes_)
                    rec += rec_label * ratio
            elif average == 'weighted':
                rec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        rec_label = 0
                    else:
                        rec_label = float(tp) / (tp + fn)
                    ratio = Counter(self.actuals)[label] / float(n)
                    rec += rec_label * ratio
            else:
                raise Exception("Invalid average input, please select macro, micro, or weighted")



        return rec

    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        if target:
            prec =  self.precision(target=target, average = average)
            rec =  self.recall(target=target, average = average)
            if prec+rec == 0:
                f1_score = 0
            else:
                f1_score =  2*prec*rec/(prec+rec)
        else:
            if self.confusion_matrix==None:
                self.confusion()

            if average == "macro":
                # f1_score_list = []
                # classes_list = self.classes_.copy()
                # for target_class in classes_list:
                #     prec =  self.precision(target=target_class, average = average)
                #     rec =  self.recall(target=target_class, average = average)
                #     if prec+rec == 0:
                #         f1_score = 0
                #     else:
                #         f1_score =  2*prec*rec/(prec+rec)
                #     f1_score_list.append(f1_score)
                # f1_score = np.average(f1_score_list)
                
                n = len(self.actuals)
                f1_score_list = []
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        rec_label = 0
                    else:
                        rec_label = float(tp) / (tp + fn)
                    rec = rec_label

                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    
                    prec = prec_label

                    if prec+rec == 0:
                        f1_score = 0
                    else:
                        f1_score =  2*prec*rec/(prec+rec)
                    f1_score_list.append(f1_score)
                
                f1_score = 0
                ratio = 1 / len(self.classes_)
                for score in f1_score_list:
                    f1_score += ratio * score

            
            elif average == "micro":
                prec =  self.precision(target=target, average = average)
                rec =  self.recall(target=target, average = average)
                if prec+rec == 0:
                    f1_score = 0
                else:
                    f1_score =  2*prec*rec/(prec+rec)
            
            elif average == "weighted":
                # f1_score_list = []
                # classes_list = self.classes_.copy()
                # for target_class in classes_list:
                #     prec =  self.precision(target=target_class, average = average)
                #     rec =  self.recall(target=target_class, average = average)
                #     if prec+rec == 0:
                #         f1_score = 0
                #     else:
                #         f1_score =  2*prec*rec/(prec+rec)
                #     f1_score_list.append(f1_score)
                # f1_score = np.average(f1_score_list)

                n = len(self.actuals)
                f1_score_list = []
                ratio_list = []
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        rec_label = 0
                    else:
                        rec_label = float(tp) / (tp + fn)
                    rec = rec_label

                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    
                    prec = prec_label

                    if prec+rec == 0:
                        f1_score = 0
                    else:
                        f1_score =  2*prec*rec/(prec+rec)
                    f1_score_list.append(f1_score)
                    ratio_list.append(Counter(self.actuals)[label] / float(n))
                
                f1_score = 0
                for i, score in enumerate(f1_score_list):
                    ratio = ratio_list[i]
                    f1_score += ratio * score


            else:
                raise Exception("Invalid average input, please select macro, micro, or weighted")


        return f1_score


    def auc(self, target):
        # compute AUC of ROC curve for each class
        # return auc = {self.classes_[i]: auc_i}, dict
        if type(self.pred_proba)==type(None):
            return None
        else:
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]
                tp = 0
                fp = 0
                fn = Counter(self.actuals)[target]
                tn = len(self.actuals) - fn
                tpr = 0
                fpr = 0
                auc_target = 0
                for i in order:
                    if self.actuals[i] == target:
                        tp = tp + 1
                        fn = fn - 1
                        tpr = tp/(tp+fn)
                    else:
                        fp = fp + 1
                        tn = tn - 1
                        pre_fpr = fpr
                        fpr = fp/(fp+tn)
                        auc_target = auc_target + tpr*(fpr - pre_fpr)
            else:
                raise Exception("Unknown target class.")

            # auc = 0
            # actuals = self.actuals
            # previous_FPR = 0
            # unique_probabilities = -np.sort(-1*np.unique(self.pred_proba[target]))
            # sorted_probabilities = -np.sort(-1*self.pred_proba[target])
            # for unique_probability in sorted_probabilities:
            #     predictions = self.pred_proba[target] >= unique_probability
            #     tp = np.sum(np.logical_and(predictions, actuals == target))
            #     tn = np.sum(np.logical_and(~predictions, actuals != target))
            #     fp = np.sum(np.logical_and(predictions, actuals != target))
            #     fn = np.sum(np.logical_and(~predictions, actuals == target))
            #     TPR = tp/(tp+fn)
            #     FPR = fp/(fp+tn)
            #     auc += TPR*(FPR - previous_FPR)
            #     print(TPR*(FPR - previous_FPR))
            #     previous_FPR = FPR
            
            # auc_target = auc
            return auc_target


