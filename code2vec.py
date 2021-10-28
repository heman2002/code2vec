from logging import NOTSET

from numpy.lib.histograms import _histogram_bin_edges_dispatcher
from vocabularies import VocabType
from config import Config
from interactive_predict import InteractivePredictor
from model_base import Code2VecModelBase
import os
import numpy as np
from extractor import Extractor

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'

def load_model_dynamically(config: Config) -> Code2VecModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}
    if config.DL_FRAMEWORK == 'tensorflow':
        from tensorflow_model import Code2VecModel
    elif config.DL_FRAMEWORK == 'keras':
        from keras_model import Code2VecModel
    return Code2VecModel(config)


if __name__ == '__main__':
    config = Config(set_defaults=True, load_from_args=True, verify=True)

    model = load_model_dynamically(config)
    config.log('Done creating code2vec model')

    if config.is_training:
        model.train()
    if config.SAVE_W2V is not None:
        model.save_word2vec_format(config.SAVE_W2V, VocabType.Token)
        config.log('Origin word vectors saved in word2vec text format in: %s' % config.SAVE_W2V)
    if config.SAVE_T2V is not None:
        model.save_word2vec_format(config.SAVE_T2V, VocabType.Target)
        config.log('Target word vectors saved in word2vec text format in: %s' % config.SAVE_T2V)
    if (config.is_testing and not config.is_training) or config.RELEASE:
        eval_results = model.evaluate()
        if eval_results is not None:
            config.log(
                str(eval_results).replace('topk', 'top{}'.format(config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION)))
    if config.PREDICT:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if config.PREDICTION_FOLDER is not None:
        predictor = InteractivePredictor(config, model)
        directory = os.fsencode(config.PREDICTION_FOLDER)
        x, y = None, None
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename=='predicted_vector.npy' or filename=='predicted_target.npy':
                continue
            temp = predictor.predictAndSave(os.path.join(config.PREDICTION_FOLDER, filename))
            if temp is None:
                continue
            if x is None:
                x = np.reshape(temp,(1, temp.shape[0]))
                y = np.array(filename)
            else:        
                x = np.vstack((x, temp))
                y = np.vstack((y, filename))

        print(x)
        print(y)
        np.save(os.path.join(config.PREDICTION_FOLDER, 'predicted_vector.npy'), x)
        np.save(os.path.join(config.PREDICTION_FOLDER, 'predicted_target.npy'), y)
    if config.JavaExtractor is not None:
        print(config.JavaExtractor)
        path_extractor = Extractor(config, jar_path=JAR_PATH, max_path_length=MAX_PATH_LENGTH, max_path_width=MAX_PATH_WIDTH)
        try:
            predict_lines, hash_to_string_dict = path_extractor.extract_paths(config.JavaExtractor)
            print('predict_lines')
            print(predict_lines)
            print('hash_to_string_dict')
            print(hash_to_string_dict)
        except ValueError as e:
            print(e)      
            
    model.close_session()
