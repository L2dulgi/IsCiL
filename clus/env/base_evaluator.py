class BaseEvaluator() :
    '''
    Base class for evaluators
        - evaluate_base : evaluate the performance of the agent
    '''
    def __init__() :
        pass

    def evaluate_base(self, data, **kwargs) :
        raise NotImplementedError