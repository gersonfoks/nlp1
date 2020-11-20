
class CallBack:

    def __init__(self):
        self.name = None
        self.results = None
        self.reset_results()

    def forward(self, out, y):
        raise NotImplementedError

    def accumulate(self):
        '''
        Accumulate the info that is gathered into one statistic. Should reset if necessary
        :return:
        '''
        raise NotImplementedError

    def reset_results(self):
        raise NotImplementedError


class AccuracyCallback(CallBack):

    def __init__(self):
        super().__init__()
        self.name = "accuracy"
        self.count = 0

    def forward(self, out, y):
        predictions = out.argmax(dim=-1).view(-1)
        self.results += (predictions == y).sum().item()
        self.count += len(y)

    def accumulate(self):
        result = self.results / self.count
        self.reset_results()
        return result

    def reset_results(self):
        self.results = 0
        self.count = 0


class ListCallback(CallBack):

    def __init__(self, callbacks):
        super().__init__()
        self.name = "listCallback"
        self.callbacks = callbacks

    def forward(self, out, y):

        for callback in self.callbacks:
            callback.forward(out, y)

    def accumulate(self):
        result = {}
        for callback in self.callbacks:
            result[callback.name] = callback.accumulate()
        self.reset_results()
        return result

    def reset_results(self):
        pass
