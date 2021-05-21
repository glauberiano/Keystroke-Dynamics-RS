import pandas as pd
import numpy as np
from adaptive_methods.adaptive_methods import AdaptiveStrategy
import IPython.display as ipd
import copy

class BiometricSystem:
    """ Classe com implementação do Sistema Biometrico de Reconhecimento (SRB)."""

    def __init__(self, detector, random_state=False):
        self.detector = detector
        self.random_state = random_state

    def enrollment(self, dataset, adaptive):
        """ Dados dos usuários são registrados no sistema. São geradas as referências biométricas iniciais de cada usuário.
        É definido o método de adaptação de cada usuário a partir da estratégia de meta-aprendizado escolhida. São definidos os
        limiares de decisão de cada usuário. """
        
        df = copy.deepcopy(dataset)

        self.users = dict()
        
        #import pdb;pdb.set_trace()
        for i, user in enumerate(df.keys()):
            ipd.clear_output(wait=True)
            print("Iniciando cadastramento de amostras biométricas no sistema:")
            print(f"Usuário: {i+1}/{len(df.keys())} ")
            train_size = df[user].shape[0] // 2
            #self.biometric_reference[user] = self.detector.train(training_data=df[user].iloc[:train_size])

            if self.detector.name == "DoubleParallel":
                pass
            else:
                self.users[user] = {
                    'biometric_reference' : self.detector.train(training_data=df[user].iloc[:train_size]),
                    'model' : self.detector,
                    'adaptive' : adaptive
                }
        return self

    def autenticate(self, genuine_user, test_stream, decision_threshold, return_scores=False):
        decision = np.mean([*decision_threshold.values()])
        
        # Faço essa cópia para não modificar a referência biométrica em testes de fluxos de dados de outros cenários.
        biometric_reference = copy.deepcopy(self.users[genuine_user]['biometric_reference'])

        if self.users[genuine_user]['adaptive']==False:
            if return_scores:
                stream_scores = test_stream.apply(lambda x: self.detector.score(x, biometric_reference.model), axis=1) 
            else:
                y_pred = test_stream.apply(lambda x: self.detector.test(x, biometric_reference.model, decision), axis=1) 

        else:
            AS = AdaptiveStrategy(detector= self.detector)
            y_pred = list()
            stream_scores = list()

            for _, features in test_stream.iterrows():
                score = self.detector.score(sample=features, user_model=biometric_reference.model)
                stream_scores.append(score)
                if score > decision:
                    y_pred.append(1)
                    biometric_reference = AS.update(strategy=self.users[genuine_user]['adaptive'], 
                                                    biometric_reference=biometric_reference, 
                                                    new_features=features)
                else:
                    y_pred.append(0)
        
        if return_scores:
            return stream_scores
        else:
            return y_pred

    def compute_metrics(self, y_true, y_pred):
        if type(y_pred) == list:
            y_pred = pd.Series(y_pred)
        y_genuine = y_pred[y_true==1]
        y_impostor = y_pred[y_true==0]

        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - ((FNMR + FMR) / 2.0)
        return FMR, FNMR, B_acc

    def compute_metrics_scores(self, y_true, y_scores, decision_threshold):
        y_genuine = [1 if y_scores[i] > decision_threshold else 0 for i, target in enumerate(y_true) if target == 1]
        y_impostor = [1 if y_scores[i] > decision_threshold else 0 for i, target in enumerate(y_true) if target == 0]

        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - ((FNMR + FMR) / 2.0)
        return FMR, FNMR, B_acc