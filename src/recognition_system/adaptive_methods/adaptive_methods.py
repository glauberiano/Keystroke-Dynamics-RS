class AdaptiveStrategy:
    ''' Métodos adaptativos.

    Parameters:
    \t trainFunction: uma função de treinamento.
    \t userModel: o modelo do usuário mais atual.
    \t newData: dado utilizado para atualizar o modelo do usuário.

    Return:
    \t Modelo do usuário atualizado.

    '''
    def __init__(self, detector):
        self.detector = detector
        
    def update(self, strategy, biometric_reference, new_features):
        #eval('self._' + strategy)()
        if strategy == 'GrowingWindow':
            new_model = self.GrowingWindow(biometric_reference, new_features)
        elif strategy == 'SlidingWindow':
            new_model = self.SlidingWindow(biometric_reference, new_features)
        elif strategy == 'DoubleParallel':
            raise Exception("Nao esta pronto ainda")
        else:
            raise Exception("Escolha uma estratégia de adaptação válida!")
        return new_model

    def GrowingWindow(self, biometric_reference, new_features):
        biometric_reference.features = biometric_reference.features.append(new_features, ignore_index=True)
        new_model = self.detector.train(training_data=biometric_reference.features)
        return new_model

    def SlidingWindow(self, biometric_reference, new_features):
        biometric_reference.features = biometric_reference.features.iloc[1:]
        biometric_reference.features = biometric_reference.features.append(new_features, ignore_index=True)
        new_model = self.detector.train(training_data =biometric_reference.features)
        return new_model

    def DoubleParallel(self, biometric_reference, new_features):
        pass