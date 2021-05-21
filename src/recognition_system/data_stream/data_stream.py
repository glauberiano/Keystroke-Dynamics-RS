from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import random
import math

class DataStream(ABC):
    """ Helper class to build data streams."""
    def __init__(self, impostor_rate, rate_external_impostor=0, len_attacks=None, random_state=None):
        self.impostor_rate = impostor_rate # taxa de impostores
        self._len_attacks = len_attacks
        self.random_state = random_state
        if rate_external_impostor > 0:
            self.external = True
            self.rate_external_impostor = rate_external_impostor
        else:
            self.external=False
        super().__init__()

    def _extract_datasets(self, genuine, internal_users, external_users):
        """ Da maneira que está, não consigo comprovar que os fluxos estão sendo criados da maneira correta. Talvez seja melhor trabalhar com DataFrames ao inves de dicionário, mantendo a coluna 'subject'. """
        genuine_samples = internal_users[genuine]
        impostor_data = [internal_users[user] for user in internal_users.keys() if (user != genuine)]
        intern_impostor_samples = pd.concat(impostor_data)
        
        n_impostor = int((len(genuine_samples) * self.impostor_rate) / (1-self.impostor_rate))
        
        if self.external:
            ext_impostor_data = [external_users[user] for user in external_users.keys()]
            ext_impostor_samples = pd.concat(ext_impostor_data)
            
            n_internal_imp = int(n_impostor * (1-self.rate_external_impostor))
            impostor_samples= intern_impostor_samples.iloc[:n_internal_imp]
            
            n_external_imp = n_impostor - n_internal_imp
            external_samples= ext_impostor_samples.iloc[:n_external_imp]
            
            impostor_samples = pd.concat([impostor_samples, external_samples], axis=0, ignore_index=True)

        else:
            impostor_samples = intern_impostor_samples.iloc[:n_impostor]
            
        return genuine_samples, impostor_samples
        
    def _extrai(self, df1):
        a = df1.values.tolist()[0]
        df1.drop(df1.index[0], inplace=True)
        return a

    @abstractmethod
    def create(self):
        pass

class Random(DataStream):
    def create(self, genuine=None, intern_data=None, extern_data=None, random_state=None):
        genuine_samples, impostor_samples = self._extract_datasets(genuine, intern_data, extern_data)
        y_true = np.concatenate((np.ones(len(genuine_samples)), np.zeros(len(impostor_samples))))
        
        random.seed(random_state)
        random.shuffle(y_true)
        datastream = list()
        
        for i in y_true:
            if i == 1:
                genuine_samples = genuine_samples.reset_index(drop=True)
                datastream.append(self._extrai(genuine_samples))
            else:
                impostor_samples = impostor_samples.reset_index(drop=True)
                datastream.append(self._extrai(impostor_samples))

        datastream = pd.DataFrame(datastream, columns=genuine_samples.columns)
        return datastream, y_true

class GenFirst(DataStream):
    def create(self, data=None, genuine=None, internal_users=None, external_users=None):
        genuine_samples, impostor_samples = self._extract_datasets(data, genuine, internal_users, external_users)
        frames = [genuine_samples, impostor_samples]
        y_true = np.concatenate((np.ones(len(genuine_samples)), np.zeros(len(impostor_samples))))
        return pd.concat(frames, ignore_index=True), y_true

class ImpFirst(DataStream):
    def create(self, data=None, genuine=None, internal_users=None, external_users=None):
        genuine_samples, impostor_samples = self._extract_datasets(data, genuine, internal_users, external_users)
        frames = [impostor_samples, genuine_samples]
        y_true = np.concatenate( np.zeros(len(impostor_samples)), np.ones(len(genuine_samples)))
        return pd.concat(frames, ignore_index=True), y_true

class SeriesAttack(DataStream):
    """ Ataque em série precisa utilizar dados de um mesmo usuário, certo?"""
    def create(self, data=None, genuine=None, internal_users=None, external_users=None):
        genuine_samples, impostor_samples = self._extract_datasets(data, genuine, internal_users, external_users)
        n_series = math.ceil(len(impostor_samples) / self._len_attacks)
        lenG = math.ceil(len(genuine_samples)/n_series)
        ds = list()
        for i in range(n_series):
            i_idx = i*self._len_attacks
            g_idx = i*lenG
            try:
                ds.append(impostor_samples[i_idx:i_idx+self._len_attacks])
                ds.append(genuine_samples[g_idx:g_idx+lenG])
            except:
                ds.append(impostor_samples[i_idx:])
                ds.append(genuine_samples[g_idx:])
        return pd.concat(ds, ignore_index=True)
