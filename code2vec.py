from logging import NOTSET
from vocabularies import VocabType
from config import Config
from interactive_predict import InteractivePredictor
from model_base import Code2VecModelBase
import os
import numpy as np

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
        x, y = [], []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename=='predicted_vector.npy' or filename=='predicted_target.npy':
                continue

            x.append(predictor.predictAndSave(filename))
            y.append(filename)

        npx = np.array(x)
        npy = np.array(y)
        print(npx)
        print(npy)
        np.save(os.path.join(config.PREDICTION_FOLDER, 'predicted_vector.npy'), npx)
        np.save(os.path.join(config.PREDICTION_FOLDER, 'predicted_target.npy'), npy)    
            
    model.close_session()
