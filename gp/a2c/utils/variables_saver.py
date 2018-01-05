import tensorflow as tf
import pickle


class VariablesSaver:
    def __init__(self, sess):
        self.__sess = sess

    def dump(self, file_name):
        output_dict = {}
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables_executed = self.__sess.run(variables)
        for i in range(len(variables)):
            output_dict[variables[i].name] = variables_executed[i]
        self.__save_obj(output_dict, file_name)

    def load(self, file_name):
        try:
            print("Loading variables...")
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            dict = self.__load_obj(file_name)
            run_list = []
            for variable in variables:
                for key, value in dict.items():
                    if key in variable.name:
                        run_list.append(tf.assign(variable, value))
            self.__sess.run(run_list)
        except:
            print("No variables exit. Skipping...\n\n")

    def __save_obj(self, obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def __load_obj(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)
