
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

sys.path.append("../../../Scripts/")
import utils as gb_utils
import adversarial

# Global Constants
images_result_path = "../Results/images/"
data_result_path = "../Results/texts/"
model_save_path = "../Saved Model/"
dataset_folder = "../Processed data/"

class TwoModel():
    def __init__(self, source_dataset_path, target_dataset_path, exclude_class, save_results = True, is_subject_ids=False):
        """
            @brief: Initialize the TwoModel class.
            @param source_dataset_path (string): path to the source dataset, target_dataset_path (string): path to the target dataset, 
            exclude_class (array): class(s) which data should excluded, for example Jump Front and Back class for the MHEALTH dataset, 
            save_results (Boolean): whether to save the results such as trained model, untargeted and targeted results,
            is_subject_ids (Boolean): whether the pickle of the source and target dataset contains the subject ids or not.
        """
        self.source_dataset_path = source_dataset_path
        self.target_dataset_path = target_dataset_path
        self.exclude_class = exclude_class
        self.__n_channels = 3 # private memebers starts with ___, protected with _ and public without any 
        self.__n_window_len = 128
        self.__n_sampling_freq = 50.0
        self.save_results = save_results

        # load the datasets and make them ready
        print("Loading Source Dataset")
        self.source_x, self.source_y = self.__load_dataset(self.source_dataset_path, is_subject_ids)
        self.s_n_classes = np.max(self.source_y) + 1
        print("N_classes: {}".format(self.s_n_classes))

        print("Loading Target Dataset")
        self.target_x, self.target_y = self.__load_dataset(self.target_dataset_path, is_subject_ids)
        self.t_n_classes = np.max(self.target_y) + 1
        print("N_classes: {}".format(self.t_n_classes))

        # Other vriables that are assigned later
        self.batch_size = 0
        self.learning_rate = 0
        self.n_epochs = 0

        # Placeholder for the source dataset
        self.s_x_train = []
        self.s_y_train = []
        self.s_y_train_hot = []

        self.s_x_val = []
        self.s_y_val = []
        self.s_y_val_hot = []

        self.s_x_test = []
        self.s_y_test = []
        self.s_y_test_hot = []

        # Placeholder for the target dataset
        self.t_x_train = []
        self.t_y_train = []
        self.t_y_train_hot = []

        self.t_x_val = []
        self.t_y_val = []
        self.t_y_val_hot = []

        self.t_x_test = []
        self.t_y_test = []
        self.t_y_test_hot = []

        # Models
        self.s_model = None
        self.t_model = None
        

    def __scale_data(self, x):
        """
            @brief: Scale given x of shape (:, 128, 3) into range (-1.0, 1.0)
            @param: x, a numpy array of shape (:, 128, 3)
            @return: scaled_x of shape (:, 128, 3)
        """ 
        t_x = x.transpose(0, 2, 1)
        t_x = t_x.reshape(-1, self.__n_channels * self.__n_window_len)
        
        scaler = MinMaxScaler((-1.0, 1.0))
        t_x = scaler.fit_transform(t_x)
        
        t_x = t_x.reshape(-1, self.__n_channels, self.__n_window_len)
        t_x = t_x.transpose(0, 2, 1)

        return t_x

    def __load_dataset(self, path, is_subject_ids=False):
        """
            @brief: Load dataset from the given path. The file is supposed to be a pickle with contents either in the
                form x, y, _ or x, y.
            @param: path: String path of the pickle file
            @return: x, y numpy arrays
        """ 
        file = open(path, "rb")
        if is_subject_ids:
            x, y, subject_ids = pickle.load(file)
            print(subject_ids)
        else:
            x, y = pickle.load(file)
        file.close()

        if len(self.exclude_class):
            print("Excluding class {}".format(self.exclude_class[0]))
            indexes = np.where(y != self.exclude_class[0])[0]
            x = x[indexes]
            y = y[indexes]

        x = self.__scale_data(x)
        print("X: {}".format(x.shape))
        print("Y: {}".format(y.shape))

        return x, y

    def prepare_for_training(self, batch_size, learning_rate, n_epochs):
        """
            @brief: Prepare the source and target dataset for training: get train, validation, and test sets and CNN models.
            @param: batch_size (int), learning_rate (float), n_epochs (int)
            @return: None
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # get the training, validation, and test sets
        print("Source")
        self.s_x_train, self.s_x_val, self.s_x_test, self.s_y_train, self.s_y_val, self.s_y_test = gb_utils.split_into_train_val_test(self.source_x, self.source_y) 
        self.s_y_train_hot = keras.utils.to_categorical(self.s_y_train, self.s_n_classes)
        self.s_y_val_hot = keras.utils.to_categorical(self.s_y_val, self.s_n_classes)
        self.s_y_test_hot = keras.utils.to_categorical(self.s_y_test, self.s_n_classes)
        
        print("Target")
        self.t_x_train, self.t_x_val, self.t_x_test, self.t_y_train, self.t_y_val, self.t_y_test = gb_utils.split_into_train_val_test(self.target_x, self.target_y)
        self.t_y_train_hot = keras.utils.to_categorical(self.t_y_train, self.t_n_classes)
        self.t_y_val_hot = keras.utils.to_categorical(self.t_y_val, self.t_n_classes)
        self.t_y_test_hot = keras.utils.to_categorical(self.t_y_test, self.t_n_classes)

        # get the model
        self.s_model = gb_utils.get_cnn_model((self.__n_window_len, self.__n_channels), self.s_n_classes, self.learning_rate)
        self.t_model = gb_utils.get_cnn_model((self.__n_window_len, self.__n_channels), self.t_n_classes, self.learning_rate)

    def train_models(self, source_model_name, target_model_name, cross_eval=True):
        """
            @brief: Train the source and target models, and evaluate them on both complete source and target datasets. Also, save the models after training.
            @param: source_model_name (string) and target_model_name (string), cross_eval (Boolean): Whether to evalue the source model on the target dataset or not and evaluate
                the target model on the source dataset or not. For some cases, it might not be feasible to do cross evaluations.
            @return: None
        """
        print("Training source model {}".format(source_model_name))
        tp_loss = gb_utils.PlotLosses()
        tp_cbs = [tp_loss]
        tp_history = self.s_model.fit(self.s_x_train, self.s_y_train_hot, batch_size=self.batch_size, epochs=self.n_epochs,
        validation_data = (self.s_x_val, self.s_y_val_hot), verbose=0, callbacks=tp_cbs)

        print("Training target model {}".format(target_model_name))
        tp_loss = gb_utils.PlotLosses()
        tp_cbs = [tp_loss]
        tp_history = self.t_model.fit(self.t_x_train, self.t_y_train_hot, batch_size=self.batch_size, epochs=self.n_epochs,
        validation_data = (self.t_x_val, self.t_y_val_hot), verbose=0, callbacks=tp_cbs)

        # save the models
        if self.save_results:
            print("Saving trained source and target models")
            print("Source model path: {}".format(model_save_path+source_model_name))
            self.s_model.save(model_save_path+source_model_name)

            print("Target model path: {}".format(model_save_path+target_model_name))
            self.t_model.save(model_save_path+target_model_name)

        # evaluate the models on the source and target datasets
        print("Performance of the source model {}".format(source_model_name))
        l, a = self.s_model.evaluate(self.source_x, keras.utils.to_categorical(self.source_y, self.s_n_classes))
        print("On Source Dataset Loss: {:.3f}, Accuracy {:.3f}".format(l, a*100))

        if cross_eval:
            l, a = self.s_model.evaluate(self.target_x, keras.utils.to_categorical(self.target_y, self.t_n_classes))
            print("On Target Dataset Loss: {:.3f}, Accuracy {:.3f}".format(l, a*100))

        print("Performance of the target model {}".format(target_model_name))
        if cross_eval:
            l, a = self.t_model.evaluate(self.source_x, keras.utils.to_categorical(self.source_y, self.s_n_classes))
            print("On Source Dataset Loss: {:.3f}, Accuracy {:.3f}".format(l, a*100))

        l, a = self.t_model.evaluate(self.target_x, keras.utils.to_categorical(self.target_y, self.t_n_classes))
        print("On Target Dataset Loss: {:.3f}, Accuracy {:.3f}".format(l, a*100))

    def load_models(self, source_model_path, target_model_path):
        """
            @brief: Given model path load the source and target models
            @param: source_model_path (string): path to the source model, and target model path (string): path to the target model
        """
        self.s_model = keras.models.load_model(model_save_path+source_model_path)
        self.t_model = keras.models.load_model(model_save_path+target_model_path)


    def adversarial_init(self, epsilons, min_value, max_value, n_iter, target_class, same_target_class = True, target_target_class=None):
        """
            @brief: prepare the models for adversarial analysis. Create variables to store the results of adversarial
                attacks.
            @param: epsilons (an array of floats): adversarial perturbation budget, min_value (float), max_value (float), n_iter (int): number of iteration for the attack methods, target_class (int)
                    same_target_class (Boolean): is the target class label for ths source and target system same. If not provide the target_target_class (int) value.
            @return: None
        """
        keras.backend.set_learning_phase(0)
        self.epsilons = epsilons
        self.min_value = min_value
        self.max_value = max_value
        self.n_iterations = n_iter
        
        self.s_target_class = target_class
        self.s_y_target = np.ones(self.s_y_test.size) * self.s_target_class
        self.s_y_target_hot = keras.utils.to_categorical(self.s_y_target, self.s_n_classes)

        if same_target_class == False:
            self.t_target_class = target_target_class
            self.t_y_target = np.ones(self.s_y_test.size) * self.t_target_class
            self.t_y_target_hot = keras.utils.to_categorical(self.t_y_target, self.t_n_classes)
        else:
            self.t_target_class = target_class
            self.t_y_target = np.ones(self.s_y_test.size) * self.t_target_class
            self.t_y_target_hot = keras.utils.to_categorical(self.t_y_target, self.t_n_classes)    

        self.s_untar_results = pd.DataFrame(columns=['Epsilon', "Attack Method", "Loss", "Accuracy", "Success Score", "Modified Success Score"])
        self.s_tar_results = pd.DataFrame(columns=['Epsilon', "Attack Method", "Loss", "Accuracy", "Success Score"])
        self.t_untar_results = pd.DataFrame(columns=['Epsilon', "Attack Method", "Loss", "Accuracy", "Success Score", "Modified Success Score"])
        self.t_tar_results = pd.DataFrame(columns=['Epsilon', "Attack Method", "Loss", "Accuracy", "Success Score"])

    def adversarial_untargeted_attacks(self, source_save_name, target_save_name):
        """
            @brief: Run untargeted adversarial attacks on the source and target models with adversarial examples computed using the source model. Five different attacks methods.
            @param: source_save_name (string): name to save the source untargeted results, target_save_name (string): name to save the target untargeted results.
            @return: None
        """
        for eps in self.epsilons:
            loss, acc = 0.0, 0.0

            # get the adversarial object of the source model for this value of epsilon
            source_adv = adversarial.AdvesarialCompute(self.s_model, self.min_value, self.max_value, self.n_iterations, eps)
            
            # compute the adversarial examples using the different methods
            fgsm_exm = source_adv.fgsm_compute(self.s_x_test, self.s_y_test_hot)
            
            biter_exm = source_adv.basic_iter_compute(self.s_x_test, self.s_y_test_hot)
            
            momentum_exm = source_adv.momentum_iterative_compute(self.s_x_test, self.s_y_test_hot)
            
            saliency_exm = source_adv.saliency_compute(self.s_x_test, self.s_y_test_hot)
            
            carlini_exm = source_adv.carlini_compute(self.s_x_test, self.s_y_test_hot)
            
            # evaluate the adversarial examples on the source and target model
            # loss, acc = self.s_model.evaluate(fgsm_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.s_model, fgsm_exm, self.s_x_test, self.s_y_test)
            self.s_untar_results = self.s_untar_results.append({"Epsilon": eps, "Attack Method": "FGSM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                    "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            
            # loss, acc = self.s_model.evaluate(biter_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.s_model, biter_exm, self.s_x_test, self.s_y_test)
            self.s_untar_results = self.s_untar_results.append({"Epsilon": eps, "Attack Method": "BIM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                    "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            
            # loss, acc = self.s_model.evaluate(momentum_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.s_model, momentum_exm, self.s_x_test, self.s_y_test)
            self.s_untar_results = self.s_untar_results.append({"Epsilon": eps, "Attack Method": "MIM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                    "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            
            # loss, acc = self.s_model.evaluate(saliency_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.s_model, saliency_exm, self.s_x_test, self.s_y_test)
            self.s_untar_results = self.s_untar_results.append({"Epsilon": eps, "Attack Method": "SMM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            # loss, acc = self.s_model.evaluate(carlini_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.s_model, carlini_exm, self.s_x_test, self.s_y_test)
            self.s_untar_results = self.s_untar_results.append({"Epsilon": eps, "Attack Method": "CW", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            ###############################################  Target System   #################################################
            
            # loss, acc = self.t_model.evaluate(fgsm_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.t_model, fgsm_exm, self.s_x_test, self.s_y_test)
            self.t_untar_results = self.t_untar_results.append({"Epsilon": eps, "Attack Method": "FGSM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                    "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            
            # loss, acc = self.t_model.evaluate(biter_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.t_model, biter_exm, self.s_x_test, self.s_y_test)
            self.t_untar_results = self.t_untar_results.append({"Epsilon": eps, "Attack Method": "BIM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                    "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            
            # loss, acc = self.t_model.evaluate(momentum_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.t_model, momentum_exm, self.s_x_test, self.s_y_test)
            self.t_untar_results = self.t_untar_results.append({"Epsilon": eps, "Attack Method": "MIM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                    "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            
            # loss, acc = self.t_model.evaluate(saliency_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.t_model, saliency_exm, self.s_x_test, self.s_y_test)
            self.t_untar_results = self.t_untar_results.append({"Epsilon": eps, "Attack Method": "SMM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
            
            # loss, acc = self.t_model.evaluate(carlini_exm, self.s_y_test_hot)
            success_score, _ = adversarial.misclassification_score(self.t_model, carlini_exm, self.s_x_test, self.s_y_test)
            self.t_untar_results = self.t_untar_results.append({"Epsilon": eps, "Attack Method": "CW", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100, "Modified Success Score": _ * 100}, ignore_index = True)
        
        # save the results
        if self.save_results:
            print("Saving the results of untargeted attacks.")
            print("Source model results at {}".format(data_result_path+source_save_name+".csv"))
            self.s_untar_results.to_csv(data_result_path+source_save_name+".csv")
            
            print("Source model results at {}".format(data_result_path+target_save_name+".csv"))
            self.t_untar_results.to_csv(data_result_path+target_save_name+".csv")
        
        
    def adversarial_targeted_attacks(self, source_save_name, target_save_name):
        """
            @brief: Run targeted adversarial attacks on the source and target models with adversarial examples computed using the source model. Five different attacks methods.
            @param: source_save_name (string): name to save the source targeted results, target_save_name (string): name to save the target targeted results.
            @return: None
        """
        for eps in self.epsilons:
            loss, acc = 0, 0
            # get the adversarial object of the source model for this value of epsilon
            source_adv = adversarial.AdvesarialCompute(self.s_model, self.min_value, self.max_value, self.n_iterations, eps)
    
            # compute the adversarial examples using the different methods
            fgsm_exm = source_adv.fgsm_compute(self.s_x_test, self.s_y_test_hot, self.s_y_target_hot)
            
            biter_exm = source_adv.basic_iter_compute(self.s_x_test, self.s_y_test_hot, self.s_y_target_hot)
            
            momentum_exm = source_adv.momentum_iterative_compute(self.s_x_test, self.s_y_test_hot, self.s_y_target_hot)
            
            saliency_exm = source_adv.saliency_compute(self.s_x_test, self.s_y_test_hot, self.s_y_target_hot)
            
            carlini_exm = source_adv.carlini_compute(self.s_x_test, self.s_y_test_hot, self.s_y_target_hot)
            
            # evaluate the adversarial examples on the trained source and target models
            # loss, acc = self.s_model.evaluate(fgsm_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.s_model, fgsm_exm, self.s_x_test, self.s_y_test, self.s_target_class)
            self.s_tar_results = self.s_tar_results.append({"Epsilon": eps, "Attack Method": "FGSM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100}, ignore_index = True)
            
            # loss, acc = self.s_model.evaluate(biter_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.s_model, biter_exm, self.s_x_test, self.s_y_test, self.s_target_class)
            self.s_tar_results = self.s_tar_results.append({"Epsilon": eps, "Attack Method": "BIM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                            "Success Score": success_score * 100}, ignore_index = True)
            
            # loss, acc = self.s_model.evaluate(momentum_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.s_model, momentum_exm, self.s_x_test, self.s_y_test, self.s_target_class)
            self.s_tar_results = self.s_tar_results.append({"Epsilon": eps, "Attack Method": "MIM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100}, ignore_index = True)
            
            # loss, acc = self.s_model.evaluate(saliency_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.s_model, saliency_exm, self.s_x_test, self.s_y_test, self.s_target_class)
            self.s_tar_results = self.s_tar_results.append({"Epsilon": eps, "Attack Method": "SMM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100}, ignore_index = True)
            
            # loss, acc = self.s_model.evaluate(carlini_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.s_model, carlini_exm, self.s_x_test, self.s_y_test, self.s_target_class)
            self.s_tar_results = self.s_tar_results.append({"Epsilon": eps, "Attack Method": "CW", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100}, ignore_index = True)
            
            ############################################ Target System ##################################################
            
            # loss, acc = self.t_model.evaluate(fgsm_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.t_model, fgsm_exm, self.s_x_test, self.s_y_test, self.t_target_class)
            self.t_tar_results = self.t_tar_results.append({"Epsilon": eps, "Attack Method": "FGSM", 
                                                                "Loss": loss, "Accuracy": acc * 100, 
                                                                "Success Score": success_score * 100}, ignore_index = True)
            
            # loss, acc = self.t_model.evaluate(biter_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.t_model, biter_exm, self.s_x_test, self.s_y_test, self.t_target_class)
            self.t_tar_results = self.t_tar_results.append({"Epsilon": eps, "Attack Method": "BIM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                            "Success Score": success_score * 100}, ignore_index = True)
            
            # loss, acc = self.t_model.evaluate(momentum_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.t_model, momentum_exm, self.s_x_test, self.s_y_test, self.t_target_class)
            self.t_tar_results = self.t_tar_results.append({"Epsilon": eps, "Attack Method": "MIM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100}, ignore_index = True)
            
            # loss, acc = self.t_model.evaluate(saliency_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.t_model, saliency_exm, self.s_x_test, self.s_y_test, self.t_target_class)
            self.t_tar_results = self.t_tar_results.append({"Epsilon": eps, "Attack Method": "SMM", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100}, ignore_index = True)
            
            # loss, acc = self.t_model.evaluate(carlini_exm, self.y_target_hot)
            success_score = adversarial.success_score(self.t_model, carlini_exm, self.s_x_test, self.s_y_test, self.t_target_class)
            self.t_tar_results = self.t_tar_results.append({"Epsilon": eps, "Attack Method": "CW", 
                                                                "Loss": loss, "Accuracy": acc * 100,
                                                                "Success Score": success_score * 100}, ignore_index = True)
            
        # save the results
        if self.save_results:
            print("Saving the results of targeted attacks.")
            print("Source model results at {}".format(data_result_path+source_save_name+".csv"))
            self.s_tar_results.to_csv(data_result_path+source_save_name+".csv")

            print("Source model results at {}".format(data_result_path+target_save_name+".csv"))
            self.t_tar_results.to_csv(data_result_path+target_save_name+".csv")
